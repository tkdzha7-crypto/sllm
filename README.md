1.해약징후
- active_check.py
   results_dir = "c:/Users/user/Documents/ml/cancellation-prediction/results"    : 이게 뭔지
- contract_query
  * customer_information_base, customer_information_master
  : 고객정보 내 신고객분류체계에서 많은 고객이 가정집으로 되어있음.
    이 부분이 영업기회에 큰 영향을 안준다면 상관없지만 그게 아니라면 문제있어 보임.
  * account_details ( CESCOEIS.dbo.TB_월별미수내역 )
  : 미수내역에서 미수구분 'Y'만 필터걸면 안되는지?
    미수구분 'N'인 건은 모두 미수금액이 0원임
  * text_log
  ( CESCOSFA.dbo.TB_SALESACTIVITY_RECORD )
  : 업무일지 내 USID를 접수일시로 보는데, 해당 컬럼은 등록일시여서
    실제 접촉한 시간은 ActDate_Start를 봐야하는게 좀 더 정확하지 않나 싶음.
    하지만 며칠 뒤 등록하는 건을 제외하면 대부분 ActDate_End 이후 등록을 바로함.
  ( CESCOEIS.dbo.TB_일정통화결과 )
  : 일정통화결과 내 통화결과 코드 값이 62(해약요구), 22(작업보류), 32(작업취소) 만 사용하였는데
    일정변경(40)의 경우는 제외한 이유가 있는지?
    그리고 통화결과 내 60(해약요구), 20(작업보류), 30(작업취소) 포함 요청 (2025년 128건 / 현재 반영된 코드는 12,040건 )
    [ 12: 작업진행 , 22: 작업보류 , 32: 작업취소 , 42: 일정변경 , 52: 미통화 , 62: 해약요구 , 99: 당월미작업 ,
      10: 일정확정 , 20: 작업보류 , 30: 작업취소 , 40: 일정변경 , 50: 미통화 , 60: 해약요구 , 70: 긴급진단 취소 ]
   ( CESCOEIS.dbo.TB_미작업사유 )
   : 미작업사유, 미작업사유2 컬럼 값을 필터하지 않은 이유가 있는지?
     [ 미작업사유 코드 - SELECT * FROM CESCOEIS.dbo.TB_UNDFCODE WHERE PARENTCODE = 'A0001',
       미작업사유2 코드 - SELECT * FROM CESCOEIS.dbo.MKTCGERL WHERE Gerlgubn = '미작업사유' ]
   ( CESCOSFA.dbo.TB_Sales_Toss )
   : TSYN(확정여부) 값이 1인 것만 필터하면 안되는지? ( 전체 약 35건 중 약 7천건이라 2% 해당하긴 함 )
     -- CTYN(계약성사여부) 값이 1인 것만 필터하는 것이나 CTYN이 반영에 대해서 의견 요청
     IPPT(감동포인트) 내용이 있는 것만 필터하면 안되는지? 활동에 건수 포함의 의미라면 전체 반영이 맞음. (작성비율은 반반)
- CSI_QUERY
  : 쿼리 중 INNER JOIN IEIS.dbo.TB_WR_TOWR t with (NOLOCK) ON s.OBJT_CD = t.TOWRNO 이 부분 제외해도 되는지
    TOWR 테이블 내 컬럼을 사용하지 않아서 불필요해 보임
