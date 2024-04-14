import math
import numpy as np
import time
import matplotlib.pyplot as plt
import sympy as sp

# sympy関数のnumpyへの変換
def numpyfy(arg,f):
    return sp.lambdify(arg, f, "numpy")

#モデル変数
x, y,z= sp.symbols('x y z')
xt,yt,zt=sp.symbols('xt yt zt')
ux, uy= sp.symbols('ux uy')
lmx,lmy,lmz=sp.symbols('lmx lmy lmz')
x_ob,y_ob,z_ob=sp.symbols('x_ob,y_ob,z_ob')

#vector化
u=sp.Matrix([ux,uy])
xs=sp.Matrix([x,y,z])
obs=sp.Matrix([x_ob,y_ob,z_ob])
xbar=sp.Matrix([x-x_ob,y-y_ob,z-z_ob])
t=sp.Matrix([xt,yt,zt])
lm=sp.Matrix([lmx,lmy,lmz])

# 移動ロボットクラス(model)
class car:
    def __init__(self,R,T):
        self.r = R/2
        self.rT = R/T
        cosz = sp.cos(z)
        sinz = sp.sin(z) 
        f=sp.Matrix([self.r*cosz*(ux+uy),self.r*sinz*(ux+uy),self.rT*(ux-uy)])
        _func=numpyfy((xs,u),f)
        self.func=lambda u,xs:_func(xs,u).flatten()    

#コントローラークラス
class controller:
    def __init__(self, car, x_ob,f=None,H=None,debug=False):
        #コントローラーのパラメータ
        self.Ts = 0.05 #制御周期
        self.ht = self.Ts 
        self.zeta = 1.0/self.Ts #安定化係数
        self.tf = 1.0 #予測ホライズンの最終値
        self.alpha = 0.5 #予測ホライズンの変化パラメータ
        self.N = 20 #予測ホライズンの分割数
        self.Time = 0.0 #時刻を入れる変数
        self.dt = 0.0 #予測ホライズンの分割幅

        #入力と状態の次元
        self.len_u = 2 #入力の次元
        self.len_x = 3 #状態変数の次元

        #コントローラーの変数
        self.u = np.zeros(self.len_u)
        self.U = np.zeros(self.len_u * self.N)
        self.x = np.zeros(self.len_x)
        self.dU = np.zeros(self.len_u * self.N)

        #目標地点
        self.x_ob = x_ob

        #操縦する車
        self.car = car
        
        #偏導関数の計算
        _func=numpyfy((xs,u),f)
        self.func=lambda xs,u:_func(xs,u).flatten()

        fu=sp.Matrix([f.diff(ux).T,f.diff(uy).T]).T
        self.Calcfu= numpyfy((xs,u),fu)
        
        Hu=sp.Matrix([H.diff(ux),H.diff(uy)])
        Hx=sp.Matrix([H.diff(x),H.diff(y),H.diff(z)])
        Ht=sp.Matrix([H.diff(xt),H.diff(yt),H.diff(zt)])
        
        #numpy関数への変換,ベクトル値関数なので出力をflattenする
        _CalcHu=numpyfy((xs,u,lm), Hu)
        _CalcHt=numpyfy((t,u), Ht)
        _CalcHx=numpyfy((xs,obs,u,lm), Hx)

        self.CalcHu=lambda u,lm,xs:_CalcHu(xs,u,lm).flatten()
        self.CalcHt=lambda t,u:_CalcHt(t,u).flatten()
        self.CalcHx=lambda x,obs,u,lm:_CalcHx(x,obs,u,lm).flatten()

        self.debug=debug

    #xの予測計算
    def Forward(self, x, U):
        X = np.zeros((self.len_x, self.N+1))
        B_all = np.zeros((self.len_x, self.len_u*self.N))

        X[:,0] = x

        for i in range(1,self.N+1):
            B= self.Calcfu(X[:,i-1],U[:,i-1])
            dx=B@U[:,i-1]
            X[:,i] = X[:,i-1] + dx*self.dt
            B_all[:,self.len_u*(i-1):self.len_u*i] = B
        return X, B_all
    
    #随伴変数の計算
    def Backward(self, X, U):
        Lambda = np.zeros((self.len_x, self.N))
        Lambda[:,self.N-1] = self.CalcHt(X[:,self.N]-self.x_ob,U[:])
    
        for i in reversed(range(self.N-1)):
            Lambda[:,i] = Lambda[:,i+1] + self.CalcHx(X[:,i+1],self.x_ob ,U[:,i+1], Lambda[:,i+1])*self.dt      

        return Lambda
  
    def CalcF(self, x, U):
        F = np.zeros(self.len_u*self.N)
        U = U.reshape(self.len_u, self.N, order='F')
        X, _ = self.Forward(x, U)
        Lambda = self.Backward(X, U)
    
        for i in range(self.N):
            F[self.len_u*i:self.len_u*(i+1)] = self.CalcHu(U[:,i], Lambda[:,i], X[:,i])

        return F

    def CGMRES_control(self):
        self.dt = (1-math.exp(-self.alpha*self.Time))*self.tf/self.N
        dx = self.func(self.x, self.u)
        Fux = self.CalcF(self.x + dx*self.ht, self.U + self.dU*self.ht)
        Fx  = self.CalcF(self.x + dx*self.ht, self.U)
        F   = self.CalcF(self.x, self.U)
        
        left = (Fux - Fx)/self.ht
        right = -self.zeta*F - (Fx - F)/self.ht
        #初期残差
        r0 = right - left
        
        #GMRESの繰り返し回数(基底ベクトルの数)
        m = self.len_u*self.N
        
        #Arnoldi法
        Vm = np.zeros((m, m+1))
        Vm[:,0] = r0/np.linalg.norm(r0)
        Hm = np.zeros((m+1,m))
               
        for i in range(m):
            Fux = self.CalcF(self.x + dx*self.ht, self.U + Vm[:,i]*self.ht)
            Av = (Fux - Fx)/self.ht

            for k in range(i+1):
               Hm[k][i] = Av.T@Vm[:,k]

            temp_vec = np.zeros(m)
            for k in range(i+1):
                temp_vec = temp_vec + Hm[k,i]*Vm[:,k]

            v_hat = Av - temp_vec

            Hm[i+1][i] = np.linalg.norm(v_hat)
            Vm[:,i+1] = v_hat/Hm[i+1][i]

        e = np.zeros(m+1)
        e[0] = 1.0
        gm_ = np.linalg.norm(r0)*e
        
        #Givens rotation
        UTMat, gm_ = self.ToUTMat(Hm, gm_, m)
        #掃き出し
        min_y = np.zeros(m)
        for i in range(m):
            min_y[i] = (gm_[i] - (UTMat[i,:]@min_y))/UTMat[i][i]

        self.dU =self.dU + Vm[:,0:m]@ min_y
        self.U = self.U + self.dU*self.ht
        self.u = self.U[0:2]

    #Givens回転
    def ToUTMat(self, Hm, gm, m):
        for i in range(m):
            nu = math.sqrt(Hm[i][i]**2 + Hm[i+1][i]**2)
            c_i = Hm[i][i]/nu
            s_i = Hm[i+1][i]/nu
            Omega = np.eye(m+1)
            Omega[i][i] = c_i
            Omega[i][i+1] = s_i
            Omega[i+1][i] = -s_i
            Omega[i+1][i+1] = c_i

            Hm = Omega@ Hm
            gm = Omega@ gm
        return Hm, gm

def test(plot=False,maxTime=20,debug=False,use_sympy=True):
    #定数
    umass=1
    Q = sp.Matrix([[100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 0]])
    R = sp.Matrix([[umass, 0],
                    [0, umass]])
    
    S = sp.Matrix([[100, 0, 0],
                    [0, 100, 0],
                    [0, 0, 0]])
    barcoef=0.15 

    umax=sp.Matrix([15,15])
    umin=-umax

    carR = 0.05
    carT = 0.2
    r = carR/2
    rT = carR/carT

    #コントローラー側の運動方程式
    f=sp.Matrix([r*sp.cos(z)*(ux+uy),r*sp.sin(z) *(ux+uy),rT*(ux-uy)])

    J=(u.T*R*u+ xbar.T*Q*xbar+t.T*S*t)/2

    lu=(umax - u).applyfunc(sp.log)+(u-umin).applyfunc(sp.log)
    barrier=sp.Matrix([lu[0]+lu[1]]) #障壁関数
    #評価関数 (Hamiltonian)
    H= J + f.T*lm  +barcoef*barrier

    x_ob = np.array([3, 2, 0]).T

    nonholo_car = car(carR,carT)
    ctrl = controller(nonholo_car, x_ob,f,H,debug=debug)
    Time = 0
    start = time.time()
    xs=[]
    us=[]

    while Time <= maxTime:
        xs.append(ctrl.x)
        us.append(ctrl.u)
        x = ctrl.x + nonholo_car.func(ctrl.u, ctrl.x)*ctrl.Ts
        ctrl.Time = Time + ctrl.Ts
        ctrl.CGMRES_control()
        Time += ctrl.Ts
        ctrl.x = x

    end = time.time()
    print("計算時間：{}[s]".format(end - start))
    xs=np.array(xs).reshape(len(xs),3)
    us=np.array(us).reshape(len(us),2)
    if(plot):
        plt.plot(xs)
        plt.show()
        plt.plot(us)
        

if __name__ == '__main__':
    test(maxTime=10,debug=False,plot=True)