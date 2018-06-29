library(tidyverse)
library(lubridate)
library(xtable)
tnx <- read_delim("Data/transactions.txt", 
                  "|", 
                  escape_double = FALSE, 
                  col_types = cols(
                    ID_txn = col_character(), 
                    ID_user = col_character(),
                    txn_time = col_datetime(format = "%Y-%m-%d %H:%M:%S")), 
                  trim_ws = TRUE) %>% filter(!is.na(price))

max_time <- max(tnx$txn_time)
tnx_summary <- tnx %>% 
  group_by(ID_user) %>%
  mutate(recency   = ((min(txn_time) %--% max(txn_time)) %/% days()),
         Tm = (max(txn_time) %--% max_time) %/% days()) %>%
  group_by(ID_user, ID_txn) %>% 
  summarise(
    count           = n(), 
    transaction_sum = sum(price), 
    recency           = mean(recency), 
    Tm         = mean(Tm))
rfm_per_user <- tnx_summary %>%
  group_by(ID_user) %>%
  summarise(
    frequency    = n(), 
    n_bought     = sum(count), 
    avg_bought   = mean(transaction_sum), 
    total_bought = sum(transaction_sum),
    recency        = mean(recency),
    Tm      = mean(Tm)
  ) %>%
  mutate(recency = recency + 365) %>%
  mutate(rate_tnx_tm = frequency/(Tm/30),
         rate_tnx_rec = frequency/(recency/30)) %>%
  mutate(
    # frequency_q     = ntile(frequency, 5),
    monetary_q      = ntile(avg_bought, 2),
    Tm_q       = ntile(max(Tm) - Tm, 2),
    frequency_q_tm = ntile(rate_tnx_tm, 2),
    frequency_q_rec = ntile(rate_tnx_rec, 2)
  )
rfm_tm <- rfm_per_user %>%
  group_by(Tm_q, frequency_q_tm, monetary_q) %>% 
  summarise(
    avg_bought   = mean(avg_bought),  
    avg_bought_n = mean(n_bought),
    count        = n(),
    Tm      = mean(Tm), 
    frequency    = mean(rate_tnx_tm), 
    monetary     = mean(avg_bought)
  )

rfm_rec<- rfm_per_user %>%
  group_by(Tm_q, frequency_q_rec, monetary_q) %>% 
  summarise(
    avg_bought   = mean(avg_bought),  
    avg_bought_n = mean(n_bought),
    count        = n(),
    Tm      = mean(Tm), 
    frequency    = mean(rate_tnx_rec), 
    monetary     = mean(avg_bought)
  )
rfm_cor <- rfm_per_user %>% 
  select(Tm, rate_tnx_rec, avg_bought) %>% 
  filter(!is.na(avg_bought)) %>% 
  cor


rfm <- rfm_rec %>% 
  arrange(desc(Tm_q), desc(frequency_q_rec), desc(monetary_q))
monetary <- rfm %>% 
  group_by(monetary_q) %>% 
  summarise(recency = mean(Tm, na.rm=TRUE),
            frequency = mean(frequency, na.rm=TRUE),
            monetary = mean(monetary, na.rm=TRUE)
            ) %>%
  filter_all(any_vars(!is.nan(.)))

Tm <- rfm %>% 
  group_by(Tm_q) %>% 
  summarise(recency = mean(Tm, na.rm=TRUE),
            frequency = mean(frequency, na.rm=TRUE),
            monetary = mean(monetary, na.rm=TRUE)
            )

frequency <- rfm %>% 
  group_by(frequency_q_rec) %>% 
  summarise(recency = mean(Tm, na.rm=TRUE),
            frequency = mean(frequency, na.rm=TRUE),
            monetary = mean(monetary, na.rm=TRUE))


xtable(rfm)

xtable(monetary)
xtable(Tm)
xtable(frequency)

sum(tnx$price, na.rm=TRUE)