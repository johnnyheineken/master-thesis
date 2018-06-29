library(tidyverse)
library(readr)
library(plotly)
m <- matrix <- read_csv("matrix.csv")

sum(is.na(already_churned$ID_user))

already_churned <- m %>% filter(last_txn_days > 365)
m <- m %>% filter(last_txn_days <= 365)

sum(already_churned$pred_1m)
already_churned %>% 
  group_by(ID_user) %>% 
  summarise(bought = sum(txn_total * avg_price_txn)) %>% 
  arrange(desc(bought))



m <- m %>% 
  mutate(month=as.integer(last_txn_days/31), 
         total = round(txn_total*avg_price_thing, 2), 
         false_neg = churn == 0 & pred_1m == 1, 
         false_pos = churn == 1 & pred_1m == 0,
         true_pos = churn == 1 & pred_1m == 1,
         true_neg = churn == 0 & pred_1m == 0,
         falses = (false_neg - false_pos))
m <- m %>% mutate(evaluation=case_when(false_neg==1 ~ 'false neg',
                                       false_pos==1 ~ 'false pos',
                                      true_pos==1 ~ 'true pos',
                                       true_neg==1 ~ 'true neg'))



pal1 <- c("#011627", '#F71735','#41EAD4',  "#FF9F1C")

m %>% plot_ly(x=~txn_total,
                y=~avg_price_txn,
                frame=~month,
                color=~evaluation,
                colors=pal1,opacity=0.7,
                marker=list(size=12),
                type = 'scatter',
                mode='markers', 
                text = ~ID_user) %>%
  layout(
    xaxis = list(
      type = "log"
    ),
    yaxis = list(
      type = "log"
    )
  )

pal1 <- c("blue", "green")

m %>% plot_ly(x=~txn_total,
              y=~avg_price_txn,
              frame=~month,
              marker = list(size = 10,
                            color = ~churn,
                            line = list(color = ~falses2,
                                        width = 3)),
              type = 'scatter',
              mode='markers',
              colors = pal1,
              text = ~ID_user) %>%
  layout(
    xaxis = list(
      type = "log"
    ),
    yaxis = list(
      type = "log"
    )
  )

test <- m %>% group_by(month, pred_1m, churn) %>% summarise(mean(total)) %>% filter(churn==1) %>% filter(pred_1m==0)
test 

m%>% summarise(sum(total)) 

abc <- m %>% mutate(hypo = if_else(true_pos==1, 1, if_else(runif(n()) > 0.05, 0, 1)))


m %>% group_by(churn) %>% summarise(a=mean(txn_total), b=mean(avg_price_txn), d=n()*0.05, g=(a+5) * (b-7e3) * d, e=sum(total), f=(e+g)/e) 
m %>% summarise(sum(total))