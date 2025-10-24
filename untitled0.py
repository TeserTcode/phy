
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/eserc/Downloads/iris/IRIS.csv")

df['species']=df['species'].astype('category').cat.codes

df=df.sample(frac=1,random_state=1234).reset_index(drop=True)

ts=int(0.5*len(df))
td=df.iloc[:ts]
sd=df.iloc[ts:]


xt=td.iloc[:, 0:4].values
yt=td['species'].values
xs=sd.iloc[:, 0:4].values
ys=sd['species'].values


scaler=StandardScaler().fit(xt)
xt=scaler.transform(xt)
xs=scaler.transform(xs)


mlp=MLPClassifier(
    hidden_layer_sizes=(10,10),
    activation='relu', 
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=0
)

# Train
mlp.fit(xt, yt)

# Test
p=mlp.predict(xs)
ac=accuracy_score(ys, p)

print(f"acc:{ac:.2}")


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
NNğQQQQQgQ[]QQQQQ}^ -   . F }QQQ   ,_,rQ}r;             r<, gO_q_Q{^QQQQQQQQQQQQQQQ
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