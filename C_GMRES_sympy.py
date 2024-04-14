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
B= sp.symbols('B')
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
    def __init__(self,use_sympy=False):
        self.R = 0.05
        self.T = 0.2
        self.r = self.R/2
        self.rT = self.R/self.T
        if(use_sympy):
            cosz = sp.cos(z)
            sinz = sp.sin(z) 
            f=sp.Matrix([self.r*cosz*(ux+uy),self.r*sinz*(ux+uy),self.rT*(ux-uy)])
            _func=numpyfy((xs,u),f)
            self.func=lambda u,xs:_func(xs,u).flatten()    
        else:
            self.func=lambda u,x:       np.array([self.r*math.cos(x[2])*(u[0]+u[1]),
                         self.r*math.sin(x[2])*(u[0]+u[1]),
                         self.rT*(u[0]-u[1])])

    
#コントローラークラス
class controller:
    def __init__(self, car, x_ob,f=None,H=None,debug=False,use_sympy=False):
        #コントローラーのパラメータ
        self.Ts = 0.05 #制御周期
        self.ht = self.Ts 
        self.zeta = 1.0/self.Ts #安定化係数
        self.tf = 1.0 #予測ホライズンの最終値
        self.alpha = 0.5 #予測ホライズンの変化パラメータ
        self.N = 20 #予測ホライズンの分割数
        self.Time = 0.0 #時刻を入れる変数
        self.dt = 0.0 #予測ホライズンの分割幅

        if(debug):
            self.baricoef=0.
            umass=0
        else:
            self.baricoef=0.15
            umass=1
        #入力と状態の次元
        self.len_u = 2 #入力の次元
        self.len_x = 3 #状態変数の次元

        #評価関数の重み
        self.Q = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        self.R = np.array([[umass, 0],
                           [0, umass]])
        self.S = np.array([[100, 0, 0],
                           [0, 100, 0],
                           [0, 0, 0]])
        
        #コントローラーの変数
        self.u = np.zeros(self.len_u)
        self.U = np.zeros(self.len_u * self.N)
        self.x = np.zeros(self.len_x)
        self.dU = np.zeros(self.len_u * self.N)

        #入力の制限
        self.umax = np.array([15,15]) #各入力の最大値
        self.umin = np.array([-15,-15]) #各入力の最小値
        
        #目標地点
        self.x_ob = x_ob

        #操縦する車
        self.car = car
        
        #偏導関数の計算
        if(use_sympy):
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

    def CGMRES_control(self):
        self.dt = (1-math.exp(-self.alpha*self.Time))*self.tf/self.N
        dx = self.func(self.x, self.u)
        if(self.debug):            
            print("dt",self.dt)
            print("x",self.x)
            print("u",self.u)
            print("dx", dx)
            print("U", self.U)
            print("dU", self.dU)
               
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
        if(self.debug):            
            print("post dU",self.dU)
            print("post U",self.U)
            
    #コントローラー側の運動方程式
    def func(self, x, u):
        cos_ = math.cos(x[2])
        sin_ = math.sin(x[2])
        return np.array([self.car.r*cos_*(u[0]+u[1]),
                         self.car.r*sin_*(u[0]+u[1]),
                         self.car.rT*(u[0]-u[1])])
    def Calcfu(self,x,u):
        cos_ = math.cos(x[2])
        sin_ = math.sin(x[2])
        return np.array([[self.car.r*cos_, self.car.r*cos_],
                      [self.car.r*sin_, self.car.r*sin_],
                      [self.car.rT, -self.car.rT]])
    #dfdx
    def Calcfx(self, x, u):
        return np.array([[0, 0, -self.car.r*math.sin(x[2])*(u[0]+u[1])],
                         [0, 0, self.car.r*math.cos(x[2])*(u[0]+u[1])],
                         [0, 0, 0]])
    
    def CalcF(self, x, U):
        F = np.zeros(self.len_u*self.N)
        if(self.debug):    
            print("CalcF preU", U)

        U = U.reshape(self.len_u, self.N, order='F')
        if(self.debug):    
            print("CalcF U", U)
        
        X, B_all = self.Forward(x, U)
        Lambda = self.Backward(X, U)
    
        for i in range(self.N):
            F[self.len_u*i:self.len_u*(i+1)] = self.CalcHu(U[:,i], Lambda[:,i], X[:,i])

        if(self.debug):    
            print("F", F)
        return F
    
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

        if(self.debug):    
            print("dt", self.dt)  
            print("B_all", B_all)   
            print("forward X", X)
        return X, B_all
    
    #随伴変数の計算
    def Backward(self, X, U):
        Lambda = np.zeros((self.len_x, self.N))
        Lambda[:,self.N-1] = self.CalcHt(X[:,self.N]-self.x_ob,U[:])
    
        for i in reversed(range(self.N-1)):
            Lambda[:,i] = Lambda[:,i+1] + self.CalcHx(X[:,i+1],self.x_ob ,U[:,i+1], Lambda[:,i+1])*self.dt      
        if(self.debug):
            print("backward λ", Lambda)

        return Lambda
    
    #dH/du
    def CalcHu_reduce(self, u, lambd, B):
        return self.R@ u + B.T@ lambd +self.baricoef*(2*u - self.umax - self.umin)/((u - self.umin)*(self.umax - u))
        
    def CalcHu(self, u, lambd, x):
        return self.R@ u  + self.Calcfu(x,u).T@ lambd +self.baricoef*(2*u - self.umax - self.umin)/((u - self.umin)*(self.umax - u))        

    #dHdx
    def CalcHx(self, x,x_ob, u, lambd):
        return self.Q@(x-x_ob) + (self.Calcfx(x,u).T@ lambd)

    def CalcHt(self, x,u):
        return self.S @ x

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
    if(use_sympy):
        if(debug):
            barcoef=0
            umass=0 
        else:
            barcoef=0.15
            umass=1
        
        #定数
        Q = sp.Matrix([[100, 0, 0],
                        [0, 100, 0],
                        [0, 0, 0]])
        R = sp.Matrix([[umass, 0],
                        [0, umass]])
        
        S = sp.Matrix([[100, 0, 0],
                        [0, 100, 0],
                        [0, 0, 0]])

        umax=sp.Matrix([15,15])
        umin=-umax
        carR = 0.05
        carT = 0.2
        r = carR/2
        rT = carR/carT

        #コントローラー側の運動方程式
        cosz = sp.cos(z)
        sinz = sp.sin(z) 
        f=sp.Matrix([r*cosz*(ux+uy),r*sinz*(ux+uy),rT*(ux-uy)])

        #評価関数 (Hamiltonian)
        J=(u.T*R*u+ xbar.T*Q*xbar+t.T*S*t)/2

        lu=(umax - u).applyfunc(sp.log)+(u-umin).applyfunc(sp.log)
        barrier=sp.Matrix([lu[0]+lu[1]])

        H= J + f.T*lm  +barcoef*barrier
    else:
        f=None
        H=None

    x_ob = np.array([3, 2, 0]).T

    nonholo_car = car(use_sympy=use_sympy)
    ctrl = controller(nonholo_car, x_ob,f,H,debug=debug,use_sympy=use_sympy)
    Time = 0
    start = time.time()
    xs=[]
    us=[]

    while Time <= maxTime:
        xs.append(ctrl.x)
        us.append(ctrl.u)
        x = ctrl.x + nonholo_car.func(ctrl.u, ctrl.x)*ctrl.Ts
        if(debug):
            print("Time",Time)
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
#    test(maxTime=0.1,debug=True,plot=False)
   #test(maxTime=10,debug=False,plot=True)
    #test(maxTime=10,debug=False,plot=True,use_sympy=False)
    test(maxTime=10,debug=False,plot=True,use_sympy=True)