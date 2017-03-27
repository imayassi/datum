select export_objects('', 'CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION')
drop table CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION2;
create table CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION2
(  

    Decision_Tree_defect_prob float,
    Decision_Tree_flag float,
    Decision_Tree_retain_prob float,
    ERT_Forest_defect_prob float,
    ERT_Forest_flag float,
    ERT_Forest_retain_prob float,
    Random_Forest_defect_prob float,
    Random_Forest_flag float,
    Random_Forest_retain_prob float,
    customer_key varchar
);

  DELETE FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION;

copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\new_customer_defection_prediction_a.txt'  
NO ESCAPE delimiter E'\t'  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
( Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_500001.txt'  
NO ESCAPE delimiter E'\t'  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
( Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_1000001.txt'  
NO ESCAPE delimiter E'\t'  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_1500001.txt'  
NO ESCAPE delimiter E'\t'  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
( Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_2000001.txt'  
NO ESCAPE delimiter E'\t'  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_2500001.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_3000001.txt' 
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"' ;
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_3500001.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_4000001.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_4500001.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_-5000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_495000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\prediction_995000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_0.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_1000000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';      
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_1500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"'; 
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_2000000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';  
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\skip_prediction_2500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';   
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_11500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';  
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_12000000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"'; 
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_12500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_13000000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_13500000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
copy CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION
(Decision_Tree_defect_prob, Decision_Tree_flag, 
 Decision_Tree_retain_prob, ERT_Forest_defect_prob, ERT_Forest_flag, ERT_Forest_retain_prob, Random_Forest_defect_prob, Random_Forest_flag, Random_Forest_retain_prob, customer_key ) 
FROM LOCAL 	'C:\Users\imayassi\Retention\import_data\loyal_prediction_14000000.txt'  
NO ESCAPE delimiter ','  skip 1 ENCLOSED BY '"';
select * from  CTG_ANALYTICS_WS. SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION2 order by random() limit 1000;

SELECT COUNT(DISTINCT CUSTOMER_KEY), COUNT(RETAINED) FROM CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_TEST_FULL WHERE TAX_YEAR=2014 AND CORE_FLAG=1 AND TTO_FLAG=1 AND CUSTOMER_DEFINITION_ADJ IN ('NEW TO TURBOTAX');

select count( a.customer_key) from CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION a inner join (select * from CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE where tax_year=2015 and CUSTOMER_DEFINITION_ADJ IN ('LOYAL 5+','LOYAL')) b on a.customer_key=b.customer_key

SELECT
CASE WHEN (RANDOM_FOREST_DEFECT_PROB*100) >95 THEN 1
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >90 AND (RANDOM_FOREST_DEFECT_PROB*100) <=95 THEN 2
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >85 AND (RANDOM_FOREST_DEFECT_PROB*100) <=90 THEN 3
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >80 AND (RANDOM_FOREST_DEFECT_PROB*100) <=85 THEN 4
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >75 AND (RANDOM_FOREST_DEFECT_PROB*100) <=80 THEN 5
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >70 AND (RANDOM_FOREST_DEFECT_PROB*100) <=75 THEN 6
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >65 AND (RANDOM_FOREST_DEFECT_PROB*100) <=70 THEN 7
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >60 AND (RANDOM_FOREST_DEFECT_PROB*100) <=65 THEN 8
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >55 AND (RANDOM_FOREST_DEFECT_PROB*100) <=60 THEN 9
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >50 AND (RANDOM_FOREST_DEFECT_PROB*100) <=55 THEN 10
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >45 AND (RANDOM_FOREST_DEFECT_PROB*100) <=50 THEN 11
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >40 AND (RANDOM_FOREST_DEFECT_PROB*100) <=45 THEN 12
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >35 AND (RANDOM_FOREST_DEFECT_PROB*100) <=40 THEN 13
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >30 AND (RANDOM_FOREST_DEFECT_PROB*100) <=35 THEN 14
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >25 AND (RANDOM_FOREST_DEFECT_PROB*100) <=30 THEN 15
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >20 AND (RANDOM_FOREST_DEFECT_PROB*100) <=25 THEN 16
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >15 AND (RANDOM_FOREST_DEFECT_PROB*100) <=20 THEN 17
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >10 AND (RANDOM_FOREST_DEFECT_PROB*100) <=15 THEN 18
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >5 AND (RANDOM_FOREST_DEFECT_PROB*100) <=10 THEN 19

ELSE 20 END AS DECILE, 

COUNT(CUSTOMER_KEY) AS COUNTS_OF_CUST
FROM CTG_ANALYTICS_WS. SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION 
GROUP BY 1;

DROP TABLE IF EXISTS X;
CREATE LOCAL TEMP TABLE X ON COMMIT PRESERVE ROWS  AS /*+direct*/

        SELECT
        DISTINCT 
        I.CUSTOMER_KEY,
        RANDOM_FOREST_DEFECT_PROB,
        ENSEMBEL_PROB,  
        CUSTOMER_DEFINITION_ADJ,      
        CASE WHEN ENSEMBEL_PROB>=0.55 THEN 1 ELSE 0 END AS FLAG,
        CASE WHEN CUSTOMER_DEFINITION_ADJ IN ('NEW TO TURBOTAX') AND ENSEMBEL_PROB>=0.5 THEN 1 
             WHEN CUSTOMER_DEFINITION_ADJ IN ('PY SKIPPER', 'PY SKIPPER - PAST SKIPPER')  AND ENSEMBEL_PROB>0.50 THEN 1 
             WHEN CUSTOMER_DEFINITION_ADJ IN ('LOYAL 5+','LOYAL','PY RETURNING- PAST SKIPPER') AND ENSEMBEL_PROB>0.5 THEN 1 
                                      
        ELSE 0 END AS ENSEMBLE_FLAG



        FROM
        (
                select
                A.CUSTOMER_KEY,
                RANDOM_FOREST_FLAG,
                CUSTOMER_DEFINITION_ADJ,
                --DECILE,
                (P_ABANDONED_FLAG1::FLOAT) AS P_ABANDONED_FLAG1 ,
                

                nvl(RANDOM_FOREST_DEFECT_PROB,0) AS RANDOM_FOREST_DEFECT_PROB ,
                CASE WHEN B.CUSTOMER_KEY IS NULL THEN I_ABANDONED
                FROM (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1) A

                LEFT JOIN (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY RANDOM_FOREST_FLAG DESC) AS RANK FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION)O WHERE RANK=1) B ON A.CUSTOMER_KEY=B.CUSTOMER_KEY
                --CASE WHEN RANDOM_FOREST_DEFECT_PROB>=0.5 OR P_ABANDONED_FLAG1_SAS::FLOAT>0.5 THEN 
                --WHERE A.CUSTOMER_DEFINITION_ADJ in ('NEW TO TURBOTAX' , 'PY SKIPPER', 'PY SKIPPER - PAST SKIPPER')
        )I 

;

select sum(I_Abandoned_flag) from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES

/*********************************************************************MODEL ERROR************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
select
SUM(CASE WHEN I_ABANDONED_FLAG=0 AND RANDOM_FOREST_FLAG=1 AND A.RETAINED IS NULL THEN 1 ELSE 0 END) AS  MODEL_ERROR, 
SUM(CASE WHEN I_ABANDONED_FLAG=1 AND RANDOM_FOREST_FLAG=0 AND A.RETAINED IS NULL THEN 1 ELSE 0 END) AS  MODEL_ERROR2 

from  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD   A
left join (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1) B on a.customer_key=b.customer_key

LEFT JOIN (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY RANDOM_FOREST_FLAG DESC) AS RANK FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION)O WHERE RANK=1) C ON A.CUSTOMER_KEY=C.CUSTOMER_KEY
WHERE A.tax_year=2015 and core_flag=1 and tto_flag=1;

DROP TABLE IF EXISTS X;
CREATE LOCAL TEMP TABLE X ON COMMIT PRESERVE ROWS  AS /*+direct*/
select * from (select * from(
select
a.customer_key,
CASE WHEN I_ABANDONED_FLAG=0 AND RANDOM_FOREST_FLAG=1 AND a.RETAINED IS NULL THEN RANDOM_FOREST_DEFECT_PROB ELSE 0 END AS  prob,
CASE WHEN I_ABANDONED_FLAG=0 AND RANDOM_FOREST_FLAG=1 AND a.RETAINED IS NULL THEN 1 ELSE 0 END AS  flag,
CASE WHEN (RANDOM_FOREST_DEFECT_PROB*100) >90 THEN 1
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >80 AND (RANDOM_FOREST_DEFECT_PROB*100) <=90 THEN 2
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >70 AND (RANDOM_FOREST_DEFECT_PROB*100) <=80 THEN 3
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >60 AND (RANDOM_FOREST_DEFECT_PROB*100) <=70 THEN 4
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >50 AND (RANDOM_FOREST_DEFECT_PROB*100) <=60 THEN 5
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >40 AND (RANDOM_FOREST_DEFECT_PROB*100) <=50 THEN 6
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >30 AND (RANDOM_FOREST_DEFECT_PROB*100) <=40 THEN 7
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >20 AND (RANDOM_FOREST_DEFECT_PROB*100) <=30 THEN 8
WHEN (RANDOM_FOREST_DEFECT_PROB*100) >10 AND (RANDOM_FOREST_DEFECT_PROB*100) <=20 THEN 9

ELSE 10 END AS DECILE
from  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD   A
left join (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1) B on a.customer_key=b.customer_key

LEFT JOIN (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY RANDOM_FOREST_FLAG DESC) AS RANK FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION)O WHERE RANK=1) C ON A.CUSTOMER_KEY=C.CUSTOMER_KEY
WHERE A.tax_year=2015 and core_flag=1 and tto_flag=1 /*and a.customer_definition_adj =('NEW TO TURBOTAX')*/ and a.order_date<=current_date-380 )i where flag=1 limit 200000)u
/*union all
select * from (select * from (
select
a.customer_key,
CASE WHEN I_ABANDONED_FLAG=1 AND RANDOM_FOREST_FLAG=0 AND a.RETAINED IS NULL THEN 1-RANDOM_FOREST_DEFECT_PROB ELSE 0 END AS  prob,

CASE WHEN I_ABANDONED_FLAG=1 AND RANDOM_FOREST_FLAG=0 AND a.RETAINED IS NULL THEN 0 ELSE 1 END AS  flag
from  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD   A
left join (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1) B on a.customer_key=b.customer_key

LEFT JOIN (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY RANDOM_FOREST_FLAG DESC) AS RANK FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION)O WHERE RANK=1) C ON A.CUSTOMER_KEY=C.CUSTOMER_KEY
WHERE A.tax_year=2015 and core_flag=1 and tto_flag=1)i where flag=0 limit 15000)j*/;

select * from x ;
select count(*) from x;
select * from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES where customer_key=8113513914;
select order_date from CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD where customer_key=657355 and tax_year=2016;
select decile, count(*) from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES a inner join x on x.customer_key=a.customer_key where flag=1 group by 1;

/*****************************************************************************pdf*********************************************************************************************************************/
/*****************************************************************************pdf*********************************************************************************************************************/
/*****************************************************************************pdf*********************************************************************************************************************/
/*****************************************************************************pdf*********************************************************************************************************************/

DROP TABLE IF EXISTS pdf;
CREATE LOCAL TEMP TABLE pdf ON COMMIT PRESERVE ROWS AS /*+DIRECT*/
select distinct 
        c.auth_id, 
        0.8123012458 as prob, 
        1 as flag

from OMT_CTG_DWH_DEV.trans_clickstream_ctg_TY16 c
join ctg_analytics_ws.product_analytics_master pam
        on c.auth_id = pam.auth_id
        and pam.tax_year = 2016
where  
        prop14 in (  'option=prepareTaxes1' -- having tax professional do my taxes
        ,       'option=prepareTaxes2' -- plan to do my taxes with another software
        
        
                ) ; 
                
ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES    ADD COLUMN RETAINED FLOAT ;
UPDATE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES r
SET I_ABANDONED_FLAG=flag, P_ABANDONED_FLAG1=prob, P_ABANDONED_FLAG0=1-prob, decile=n.decile
FROM x N WHERE r.customer_key = n.customer_key; 

UPDATE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES r
SET retained=(case when b.retained is not null then 1 else 0 end)
FROM CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD b
WHERE r.customer_key = b.customer_key AND b.TAX_YEAR=2015 AND TTO_FLAG=1; 

SELECT sum(i_abandoned_flag) FROM  CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES;

SELECT COUNT(*) FROM  CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES;
SELECT sum(risk_flag) FROM  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE where tax_year=2015;

UPDATE CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE r
SET RISK_FLAG=flag
FROM x n WHERE r.customer_key = n.customer_key AND R.TAX_YEAR=2015 AND TTO_FLAG=1; 
UPDATE CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_TEST_FULL r
SET RISK_FLAG=flag
FROM x n WHERE r.customer_key = n.customer_key AND R.TAX_YEAR=2015 AND TTO_FLAG=1;
UPDATE CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD r
SET RISK_FLAG=flag
FROM pdf n WHERE r.auth_id = n.auth_id AND R.TAX_YEAR=2015 AND TTO_FLAG=1;

select decile, P_ABANDONED_FLAG1, I_abandoned_flag from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES;

select sum(risk_flag) from CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD where tax_year=2015;
    


/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
select abandoned_segment, sum(i_abandoned_flag) from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES group by 1


select
--decile,
I_ABANDONED_FLAG,
count(distinct retained) as retained,
count(distinct auth_not_complete) as anc,
sum(defected) as defected

from (SELECT DISTINCT A.CUSTOMER_KEY,I_ABANDONED_FLAG  FROM CTG_ANALYTICS_WS.SM_TY16_DEFECTION_MODEL_PYTHON_PREDICTION A INNER JOIN CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES B ON A.CUSTOMER_KEY=B.CUSTOMER_KEY)  a
left join CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD b on a.customer_key=b.customer_key and b.tax_year=2015 and core_flag=1 and tto_flag=1 group by 1;

select
--decile,
RANDOM_FOREST_FLAG,
count(distinct retained) as retained,
count(distinct auth_not_complete) as anc,
sum(defected) as defected

from (SELECT DISTINCT CUSTOMER_KEY,I_ABANDONED_FLAG,RANDOM_FOREST_FLAG  FROM  CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES B)  a
left join CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD b on a.customer_key=b.customer_key and b.tax_year=2015 and core_flag=1 and tto_flag=1 group by 1;


select
--decile,
risk_flag,
count(distinct retained) as retained,
count(distinct auth_not_complete) as anc,
sum(defected) as defected

from CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD b where b.tax_year=2015 and core_flag=1 and tto_flag=1 group by 1;


 UPDATE CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_YTD r
   SET RISK_FLAG=I_ABANDONED_FLAG
   FROM (SELECT * FROM (SELECT CUSTOMER_KEY, I_ABANDONED_FLAG::INT AS  I_ABANDONED_FLAG, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1)n
   WHERE r.CUSTOMER_KEY = n.CUSTOMER_KEY AND R.TAX_YEAR=2015 AND TTO_FLAG=1 AND CORE_FLAG=1; 


SELECT COUNT(CUSTOMER_KEY), COUNT(DISTINCT CUSTOMER_KEY) FROM
(SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1)B

REVOKE ALL PRIVILEGES ON TABLE  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE_TEST_FULL from public 



SELECT AUTH_ID FROM AGG_MOD_TY10 WHERE CORE_FLAG<>1 AND COMPLETED_FLAG=1 
















SELECT DISTINCT decile FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES
ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES DROP COLUMN I_ABANDONED_FLAG;
ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES DROP COLUMN P_ABANDONED_FLAG1;

ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES RENAME I_ABANDONED_FLAG_SAS to I_ABANDONED_FLAG;
ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES RENAME P_ABANDONED_FLAG1_SAS to P_ABANDONED_FLAG1;
ALTER TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES RENAME decile_SAS to decile;

DROP TABLE IF EXISTS Y;
CREATE LOCAL TEMP TABLE Y ON COMMIT PRESERVE ROWS  AS /*+direct*/
SELECT A.*, ENSEMBLE_FLAG AS I_ABANDONED_FLAG2,ENSEMBEL_PROB AS P_ABANDONED_FLAG12,
CASE WHEN (ENSEMBEL_PROB*100) >90 THEN 1
WHEN (ENSEMBEL_PROB*100) >80 AND (ENSEMBEL_PROB*100) <=90 THEN 2
WHEN (ENSEMBEL_PROB*100) >70 AND (ENSEMBEL_PROB*100) <=80 THEN 3
WHEN (ENSEMBEL_PROB*100) >60 AND (ENSEMBEL_PROB*100) <=70 THEN 4
WHEN (ENSEMBEL_PROB*100) >50 AND (ENSEMBEL_PROB*100) <=60 THEN 5
WHEN (ENSEMBEL_PROB*100) >40 AND (ENSEMBEL_PROB*100) <=50 THEN 6
WHEN (ENSEMBEL_PROB*100) >30 AND (ENSEMBEL_PROB*100) <=40 THEN 7
WHEN (ENSEMBEL_PROB*100) >20 AND (ENSEMBEL_PROB*100) <=30 THEN 8
WHEN (ENSEMBEL_PROB*100) >10 AND (ENSEMBEL_PROB*100) <=20 THEN 9

ELSE 10 END AS DECILE 


FROM  (SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_KEY ORDER BY I_ABANDONED_FLAG_SAS DESC) AS RANKs FROM CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES)O WHERE RANKs=1) A
LEFT JOIN X  ON X.CUSTOMER_KEY=A.CUSTOMER_KEY;


SELECT SEASON_PART, I_ABANDONED_FLAG, COUNT(A.CUSTOMER_KEY)  FROM CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE  A

LEFT JOIN CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES B ON A.CUSTOMER_KEY=B.CUSTOMER_KEY 

WHERE A.CUSTOMER_KEY=2015 AND CORE_FLAG=1 AND TTO_FLAG=1

GROUP BY 1,2;

SELECT RISK_FLAG,SEASON_PART,MAX(ORDER_DATE), COUNT(*) FROM CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE WHERE TAX_YEAR=2015 GROUP BY 1,2 ORDER BY 3 ;



DROP TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES;
CREATE TABLE CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES AS/*DIRECT*/
SELECT * FROM Y;

select count(distinct a.customer_key) from CTG_ANALYTICS_WS.SM_MODEL_TY16_SCORES  a
left join  CTG_ANALYTICS_WS.SM_CUSTOMER_RETENTION_BASE b on a.customer_key=b.customer_key where b.customer_key is null;