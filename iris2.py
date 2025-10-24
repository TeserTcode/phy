# -*- coding: utf-8 -*-

"""
RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRdsdfsdfgrrrrrrhhhh
kjshkjshkjshksjhkjshkjhskjhskjhskjhskjhskjshkjshkjhskjhs

##### ####  #####  #### 
  #   #   #   #   #     
  #   ####    #    ###  
  #   #  #    #       # 
##### #   # ##### ####  

"""
"""

df:main dataframe
ts:training set
td:training data
sd:testing data
xt:training input 
yt:training output
xs:testing input 
ys:testing output
l:lengths format:[inpt,hid...hid,outp]
w:weights format
b:biases
lr:learning rate
N:#of epoch
a:activation values
e:error
d:delta list   
dd:delta
    ///testing///
at:testing activation 
p:test results
t:test truth
ac:accuracy
   //functions///
F: activation function
Fp: F':derivative of the activation function

"""

import pandas as pd
import numpy as np




class math:
    def __init__(self, mode="swish", alpha=0.01):
        self.mode=mode
        self.alpha=alpha

    def swish(self,x):#swish(x)=x/(1+e^-Bx) swish1 here is B=1
        return x/(1+np.exp(-x))
    
    def swishd(self,x):#ddx swish(x) 0.5+(x+sinh)/(4cosh(x/2)^2)
        return 0.5+(x+np.sinh(x))/(4*(np.cosh(x/2)**2))
    
    def softmax(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)

    def relu(self,x):
        return np.maximum(0,x)

    def relud(self,x):
        return (x>0).astype(float)

    def sigm(self,x):
        return 1/(1+np.exp(-x))

    def sigmd(self,x):
        s=self.sigm(x)
        return s*(1-s)

    def lrelu(self,x):
        return np.where(x>0,x,self.alpha*x)

    def lrelud(self,x):
        return np.where(x>0,1,self.alpha)

    def F(self, x):
        if self.mode == "swish":
            return self.swish(x)
        elif self.mode == "relu":
            return self.relu(x)
        elif self.mode == "sigm":
            return self.sigm(x)
        elif self.mode == "lrelu":
            return self.lrelu(x)
        else:
            raise ValueError("F")

    def Fd(self, x):
        if self.mode == "swish":
            return self.swishd(x)
        elif self.mode == "relu":
            return self.relud(x)
        elif self.mode == "sigm":
            return self.sigmd(x)
        elif self.mode == "lrelu":
            return self.lrelud(x)
        else:
            raise ValueError("d/dxF")



class NNW:
    def __init__(self,layers,lr=0.01,activation="lrelu",winit="he",seed=0):
        self.l=layers
        self.lr=lr
        np.random.seed(seed)
        self.winit = winit.lower()
        self.w=[self.weightinit(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        self.b=[np.zeros((1,layers[i+1])) for i in range(len(layers)-1)]
        self.act=math(mode=activation)

    def weightinit(self,i,o):
        if self.winit=="he":
            return np.random.randn(i,o)*np.sqrt(2.0/i)
        elif self.winit=="glorot":
            l=np.sqrt(6.0/(i+o))
            return np.random.uniform(-l,l,(i,o))
        elif self.winit=="lecun":
            l=np.sqrt(3.0/i)
            return np.random.uniform(-l,l,(i,o))
        else:                             
            return np.random.randn(i,o)*0.1
        
    def train(self, xt, yt, N=10000):
        for n in range(N):
            a=self.forward(xt)
            e=a[-1]-yt
            d=[e*self.act.Fd(a[-1])]
            for i in reversed(range(len(self.w)-1)):
                dd=np.dot(d[0],self.w[i+1].T)*self.act.Fd(a[i+1])
                d.insert(0, dd)
            for i in range(len(self.w)):
                self.w[i]-=self.lr*np.dot(a[i].T,d[i])
                self.b[i]-=self.lr*np.sum(d[i],axis=0,keepdims=True)
    
    def forward(self, x):
        a=[x]
        for i in range(len(self.w)-1):
            a.append(self.act.F(np.dot(a[-1],self.w[i])+self.b[i]))
        a.append(self.act.softmax(np.dot(a[-1],self.w[-1])+self.b[-1]))
        return a
    
    def test(self, xs, ys):
        a=self.forward(xs)[-1]
        p=np.argmax(a,axis=1)
        t=np.argmax(ys,axis=1)
        acc=np.mean(p == t)
        print(f"Accuracy: {acc:.2f} ({np.sum(p==t)}/{len(xs)})")
        return acc



df=pd.read_csv("C:/Users/eserc/Downloads/iris/IRIS.csv")
df['species']=df['species'].astype('category').cat.codes
#species to numeric 012 

df=df.sample(frac=1,random_state=43423).reset_index(drop=True)
ts=int(0.5*len(df)) # split into training (t) and testing (s)
td=df.iloc[:ts]
sd=df.iloc[ts:]

xt=td.iloc[:,0:4].values
yt=pd.get_dummies(td['species']).values
xs=sd.iloc[:,0:4].values
ys=pd.get_dummies(sd['species']).values
#normalization
xs=(xs-xt.mean(axis=0))/xt.std(axis=0)
xt=(xt-xt.mean(axis=0))/xt.std(axis=0)


net=NNW([4,8,8,3],lr=0.01,activation="lrelu",winit="lecun",seed=0)
net.train(xt,yt,N=100)

net.test(xs, ys)

'''
QQNNğQQQQN_       JQQQQQQNNQQQjQQQQQQQQQQQQQQQQQQQQğ QQQgrQQQQQ}QQQQQQğ ;  ğQQQQgUQ
QQQNNQQQQQQ        QQQQQğQQNQğQQ})ŞQQQQQQQQQQQQQQQNQg:]QQ,QQQQQgQQQQQpB ; _QQQQğQgr
QQNQQQQQQQQğ        QQQQQNQQQQQQQQQjŞQQQğQQNQQQQQQQQQgJQQg:QQQQQQQQQQQF : ğQQQQQQQ}
QQQNNNNQQQQQg       JQQQQQQQQQQgUQQQğ;rQQQQNQQQQQQQQQQ_SQQ}ŞQQQQQQNQ]Q ,  QQQQQQQQQ
QQQQNNNNQQQQQ_       QQQQQQQQQğQg JQggQgUQQQQQQNQQQQQQQ,QQQrQQQQQQQQQ) ; ,QQQQQQQQQ
QQQQQQNNNQQQQQ_       QQQQQğQQQQQğ, QQQQp}UQQQQQQQQQQQQğ QQ}rQQQQQQQQ:~: gQQQQQQğQğ
QQQQQQNNNNQQQQğ       :QğQğQQQQQQQğ  QQQ]UpgHQQQQQğQNNQQg QQ,QQQQQQQQ ;: QQQQQQQQQQ
QQQQQNNNNNQQQQQğ       MNQQQQQQQQQQğ:rQH ,,^gŞQQHMQ_,,*HNg:Q}rQğQQQQÇ ::_QQQQQQQQQQ
QQQQQQNNQQNQQQQQ_       QQQQQQQQQQQQ};r;;};;Q}_]gQQQQQ};_: rQ;QQQ]QQ:   ğQQQQQQQQQQ
NQQQQQQQQNQQQQQQQ_      :QQQQQQQQQQQQ;;}QQQ}Q;QQQQQQQQQ}r,  r]~QQQQğ ;: QğQQQQQQQQQ
QNNQQQQQğğQQQQQQNN_      Jqq}QQ}qppqq;çQQQQQQQQQQQQQQQQQOr ; Q;QQQQF;; }QNNQQQQQQQQ
QQNQQQQQNNQQğğQQQQh       =;r}q;;;;}QQ_^QQQQQQpUgQQQQQO}_~_Qg g~QQQ;};_ğ}QQQQQQQQQQ
QQQNQQQQQNNQQQNQrr;,       ;;}}}}]QQQQğ :QQQQU ]QQQQQÇ;Q]gQQQ_:;ŞQÇ;;;QQ}QQQNQQQQQQ
NNQQNQQQQQQQNQQQg}}g_      rQgğQğQQQQQQ_  ^    _ JQU)QUQQQğQQQ,rrQ;};;ğQğQQQQQQQNQQ
HNNQQğQQQQQQQQQgQQQQN_      QrJHQQQQQQQQ    ,;;    _gQ_SQQQQQQQ,:q;};QQQQgQQQQQQQğğ
}}]HQUQQNQNQQQQNQQQNQğ_      ,},_  QQQQQg:   :;  _ QQQQg;QğQQQQg~;;;;gQQQQpQQğQQQQQ
Qpp}}}UHHQQQQNQQQQQQQQQ_     }}};~, ^HQQQ::  ,: .  QQgQQQ_HQQQQQg;};;QQQQQggQQQQğQğ
ğQgg}qp};rUHQQQQQQQQQQQQ_    ]Q}}};;  ğQQ,:  ::  F ğQNğQQQ_SQQQQp}};QQQğğQQNQgUQQQğ
QQQNQgg;rq};;rrUQQQQQQQpQ    pQQ}Q};; JQğr        ]QQQUHQQQgSQQQQ}};QQQNQQQQQQğgQQQ
QğQQQQQğ;;_;rq;;;rrHQ}}}pQ  ,}QQ}Qr~:  rp       gQQH      ^^:^QQQQ};QQQQNQQQQQQQğQN
gQQQQQQQgQQQg}_rr;, ::rr}pg ;}QQ]QQr;  ;        QQH           ^QQQQ}rQQQQQNQQQQQQNN
QQQQQğQQQ}QQQ}Qg__ :~,  :rU r}}Q];Q:;, }        QU   ,;; :  ,  ]QQgQ;rQQQQQQNQNQQQQ
QQpğğQQQQğQQQğQQQQğ_, ::,   ;U]QQ};,;  ğ        g    ::r     : QQQQQ],QQQQQQQQQNQQQ
QNğQQQgQQğ;QğjQQQgQQ]}_, :: Ş}r}}Q;;ç  ğ        F ;         , }_rUQNQQ;QQQQQQQQQQQQ
QQQjgQQQQğ}QQğüğQQQQ}}QQ},  ;rŞ_ŞQ]};  [   ,   ; ,;,;,;       QQ]_^QQQgrQQQQQQQQQğQ
QrQQQQQQQNQQQQ;QQQQQQ}QQ}};,rrp};QQp;  [   ;   r:;;: ;   :, ,_QQğQg:QQQ}QQQQQQQQQQQ
QQgQQQQğQQQQQ]]QQQQQQQQQ}}}} ;: r}QgQ  [  ;;  ,  :};: _;;r  ;}QQQQQgrQQQ;QQQQQQQQQQ
ğUQQQggQQQQQQQQpQQQQQQQQ}}}},  ~}r;Q]  ;  ;;  ç  _::,,;;,: ~:QQg]HQQgQQQgrQQQQQQQQQ
}pNQQQQgQQQQQQQgQQQQQQQQ}QQ}},    rUg] [  Q   :; q;}};:^  ::]QgQQggQN}QQQ}QQQQQQQQQ
ğgQ]QQQQQQQğQpr^^JQ]QQQQ}QQQQ}}_    UQ_[ ,Q  _ ;}pQ]r     r:]QQQQQNgQQ]QQQ;NNQNNQQQ
UQ]Q_QQQQQğQp,,:;; ^QQQQ}}pQQQQ}Qgg__QQÇ ]Q  |}p}Qr;~, ::r;:QgQQQQNQQQQpQQgQNNNQNQN
pğQgjqQQQQQQr;;;}]}  ü]}}}}QOr^^  ^^^^Hğ_QQ  QQür::;:,;rr;:]QQQQQQQQQQggQNQ]QQQQQQN
QQQQQ}QQQQQğ}}}QQQp:  ^S} r            JgQQ_gQ)    :,;;;: ]QQQQQQQğQQQNğgQQQQQNNNQN
QQQQQQgpQQQF}}pQ]r_^,     :         ,}QQQQQÇ  ^U.    : :_QQQQQQQQğQQQQQQQQQQQQQQNNQ
QQgğQgQ]QQQ pQQQQp]               ,;pqUUQQQp_     Ug___gQQQQQQQQQQQQQQQQQQQQQQQQQNQ
ğğQQQQQQQQQ QQ]QjQ~S~__{        ::       HOp},      QQQQQQQQQQQQQQQQQQNQQQğQQQQQQNQ
QQQQQQQQQQğ]QQQQQgu    :               ,][  r;:      QOQQQQQQQQQQQQQQQQQQQQgQQQQQQQ
NQQğQQğQQg[QQQQQpQ,~                _uC QQ_   :       _QQQQHHQQQQQQQQQQQQQQQQQQQQQQ
QQQQNQQQQQ[ğQQQQQQ,_    .       _      ;q}r           rH^HF _  ^HQQNQQQQQQQQQQQQgQQ
ğQQQQQQQQQğQQQQQQp;^   :.r_  __}    _,~};               ; :gr,_q,;HQQNQQQQQNNQNQQQQ
NNğQQQQQgQ[]QQQQQ}^-  . F }QQQ   ,_,rQ}r;             r<, gO_q_Q{^QQQQQQQQQQQQQQQ
gNNQğQQQQgğQQQQpQ__U,      ;pQQ]   ,_:;Qrrr.           _r/_^_gQQgQQ}^ğQğQQQQQQQQQQğ
QQHQQğQgggQrQQQQQg^r   /   }}QQQ  :r : p:=Ş,  }r _     JUu d^ggQQQQp;JQQNQQQQQQQQQN
gğğgğQQQQQQuQQQQQQQ__,_ }  }}QQQ, ~ , ;: ;S,: ; gQ} :},S}uOuQQQQQQQQq QQQQQQQQQQQQQ
QQQQQQQNQQQ,QQQQ]QQQO{Q~Ş: }}QQQ{ ;,  ~~  ~;;  gQQ  u^ ~^{^r^QQQQQQQ];QQQQQQQQQQQQQ
QQQQQQQQQQQ_Q]]QQQQQQ(Quu  }}QQQQ  ~  ;;  : : ğQQQ ~{QS S SrjHQQQQQQQ}}QQQQQQQQQQQQ
ŞQQQQQğQQQQğQ;}}r]QQQrQuu }}}}QQ]} : ::    ;  QQQQ, ^:,: Sr_jQQQQQQQQ}rQğQQQQQQQgQQ
,QQQQQğQNNN^Q};;QrQQQrU_ ~}}}QQQQQ;:      ;   QQQQ{.]_:,. ^_QgQQQQQQQ};QNQQQQÇgQQQQ
;,QQQQNQNNNqQ);;r}Q}Q;    }}}}QQQQp :   :    _QQQQ{ P[{[,_ ^jQQğQQQ]}çğQQQQğjgğQgQQ
;;rQQQNNQğQQQ{^;;;Qrr:   }}}}}QQQQp,::      _QQQQQ:  ([]rQg.QQQQQQQQ}~QNQQH_ğQgQQQ}
;;;rQQQQQQQQQp,r:qr;  : ]}}}}}}QQQQQ  ~~_ğQQQQQQQQ  /gj]_gQgQQğQQQQ]r_QQQQQQ]HQpQQQ
;;;;;QQQNNQQQQQ_: ;   _ğQ}}}}}QQQQQQ   ~QQQQQQQQQQ  qQQgQQQQğQQQQQQp:QQH}U}}}}}QQQQ
:;;;;;QQQQNQQQQQ_   _gQQQ}}}}}}QQQQQ{  gQQQQQQQQQQ__QQQQQQQQQQQQgpQggQÇ}}};;}QQQQQQ
g;;;;;;QQQNQQQQQQ]ggQQQQQ}}}}}}QQQQQQ_gQQQQQQQQQQQQQQQQQQQQQQQQQppOrr}p}}}}}}QQQQQQ
pp;;;;;,QQQQQQQQQQQQQQQQQQ}}}}}QQQQQQQQQQQQQQQpQQQrQQQQQQQQQQQQq;;;;Q]}}}}}QQQQQQQQ
}qp;;;;;QQQQQQQQQQQQQQQQQQ}}}}}QQQQQQQQQQQğQQQ}QQQ,QQQQQQQQQQQ]r}r}Q]};}QQQQQQQQQQQ
;;;q;;QQQQQ}QQQQQQQQQQQQQQQ}}}}QQQQQQQQQQğQQQQ}QQQ,QQQQQQQQQ}q}r;pQq;}}QQQQQQQQQQQU
};r}QQQQQQQQQQQQQQQQQQQQQQQ}}Q}QQQQQQQQQQQQQQQ}QQQ],QQpQQp]p}};pQQ}r}}QQQQQpŞ)}QgQg
};QQQQQQQQQQQQQQQQQQQQQQQQQ}}}}QQpQQQQQQQQQQQQ}gQQQQ};rqOq};;;}}}};;}}rpggggğQQQğğQ
QQQQQQQQQQQQQQQQQQQQQQQQQQQ}}}}}}}QQQQg]QOQQQ};QQQ}p}p}}p];;;}}}}};;};}}jHQNNQQğQQQ
QQQQQQQQQQQQQQQQQQQQQQQQQQ]}}}}}}QQQQQQğgQQQQQQNQQQpQ}}}r_}}}}}}}};q}}}}}p;HQQQQQQQ
QQQQQQQQQQQQQQQQQQQQQQQQQğg}}}}}QQ}QQQQQQQQQğğQQQQ}}}pr;}}]q}}}};r}}}}}}}}}Ş)NQNNNQ
                                                      
#####         #                                         #                #                
  #   ####         ####       #   #  ###  ####   ####        ###   ###   #     ###  ####  
  #   #       #   ###         #   # #   # #     ###     #   #     #   #  #    #   # #     
  #   #       #      ##        # #  ##### #        ##   #   #     #   #  #    #   # #     
##### #       #   ####          #    ###  #     ####    #    ###   ###    ##   ###  #     
'''
        
        
        
        