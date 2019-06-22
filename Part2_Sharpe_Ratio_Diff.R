
# clear invironment
rm(list=ls())

library(extrafont)
#font_import(paths = NULL, recursive = TRUE, prompt = TRUE,pattern = NULL)

library(PeerPerformance)
library(SharpeR)
library(tidyverse)
library(ggpubr)
library(cowplot)
library(colorspace)
#loadfonts(device = "win")

# save plot
save_plots<-"C:/Users/Pieter-Jan/Documents/Factor_Crashes/Thesis/Images"
# path for plot style functions
Helpers <- "C:/Users/Pieter-Jan/Documents/Factor_Crashes/Code"
setwd(Helpers)

source("Part2_Helper_functions.R")
path1 <- "C:/Users/Pieter-Jan/Documents/Factor_Crashes/Code/Simulation/VM_Portfolios/in_sample/Monthly_Simple"
path2 <- "C:/Users/Pieter-Jan/Documents/Factor_Crashes/Code/Simulation/VM_Portfolios/out_of_sample/Monthly_Simple"
setwd(path1)


#path = "C:/Users/Pieter-Jan/Documents/Factor_Crashes/Code"
# functions paper wolf
# load("Sharpe.RData")

# get files from python
files<-list.files(path = path1)
# read in 1 file to get the portfolio names
df_SR<-read.csv(files[1])
original_portfolios<-c(colnames(df_SR)[2:5])
VM_portfolios<-c(colnames(df_SR)[6:9])

# parameter sharpe ratio test
control<-list(type = 1,hac=TRUE,bBoot = 0)

# just quick test
test<-read.csv(files[2])
# ?sharpeTesting
sharpeTesting(test$Mkt.RF, test$VM_Mkt, control = control)$pval[1]*100


# perfrom sharpe ratio test
df_sr_all<-test_Sharpe_ratio(files=files,
                  original_portfolios=original_portfolios,
                  VM_portfolios=VM_portfolios,
                  ctr=control)



a <- 19
b <- 29
c <- 34
d <- 18
(RR <- a * (c + d)/(c * (a + b)))


n1 <- 53
n2 <- 47
LL <- log(RR) - qnorm(0.975)* sqrt(c/(a * n1) + d/(b * n2))
UL <- log(RR) + qnorm(0.975)* sqrt(c/(a * n1) + d/(b * n2))
c(LL, UL)

(z <- log(RR)/(sqrt(c/(a * n1) + d/(b * n2))))
## [1] -2.316
2 * (1 - pnorm(abs(z)))



# global parameters
#fonts()
FONTSTYLE<-"Comic Sans MS"
FONTSIZE<-12
C1="#009E73" 
C2="#D55E00"
XLIM <-c(-0.11,0.21) #monthly c(-0.11,0.21) #daily c(-0.02,0.05)
VARS<- c("leverage","period","portfolio","pval","tstat","se",
         "level","conf.low","conf.high",
         "dsharpe")

# redefine periods
TIME_PERIODS<-rep(c("1927-1949","Full period","1950-1974", 
                    "1975-1999","2000-2019"),times=2)
# confidence level
CI<-c(0.95)


# get the portfolios with the confidence level
out<-CI_portfolio(df=df_sr_all,vars=VARS,CI=c(0.95))


# get sharpe ratio for the full sample period
out%>%
  filter(period=="1927-2019")%>%
  filter(leverage=="ul")

#============================================================#
# market portfolio
#============================================================#

df_market<-out %>%
  filter(portfolio == "Mkt.RF")

# redefine time periods
df_market$period<-TIME_PERIODS

df_market[c("leverage","period","pval","conf.low","conf.high")]

# make plot
p_market<-CI_plot_2(df=df_market, xlims=XLIM,
          title="Market Portfolio (Mkt-RF)",
          col1=C1, col2=C2,
          fontstyle=FONTSTYLE,
          fontsize = FONTSIZE);p_market

#============================================================#
# value portfolio
#============================================================#

df_hml<-out %>%
  filter(portfolio == "HML")

# redefine time periods
df_hml$period<-TIME_PERIODS
# make plot
p_hml<-CI_plot_2(df=df_hml, xlims=XLIM,
                    title="Value Portfolio (HML)",
                    col1=C1, col2=C2,
                    fontstyle=FONTSTYLE,
                    fontsize = FONTSIZE);p_hml


#============================================================#
# momentum portfolio
#============================================================#

df_wml<-out %>%
  filter(portfolio == "WML")

# redefine time periods
df_wml$period<-TIME_PERIODS
# make plot
p_wml<-CI_plot_2(df=df_wml, xlims=XLIM,
                 title="Momentum Portfolio (WML)",
                 col1=C1, col2=C2,
                 fontstyle=FONTSTYLE,
                 fontsize = FONTSIZE);p_wml


#============================================================#
# value-momentum portfolio
#============================================================#

df_hml_wml<-out %>%
  filter(portfolio == "HML.WML")
# redefine time periods
df_hml_wml$period<-TIME_PERIODS
# make plot
p_hml_wml<-CI_plot_2(df=df_hml_wml, xlims=XLIM,
                       title="Value-Momentum Portfolio (HML-WML)",
                       col1=C1, col2=C2,
                       fontstyle=FONTSTYLE,
                       fontsize = FONTSIZE);p_hml_wml


round_df <- function(x, digits) {
  # round all numeric variables
  # x: data frame 
  # digits: number of digits to round
  numeric_columns <- sapply(x, mode) == 'numeric'
  x[numeric_columns] <-  signif(x[numeric_columns], digits)
  x
}

round_df(df_hml_wml[c("leverage","period","pval",
                  "tstat","conf.low","conf.high","dsharpe")], 5)

#============================================================#
# make a grid plot
#============================================================#


legend_b <- get_legend(p_market + theme(legend.position="bottom"))

noleg<-theme(legend.position="none")
p_grid<-plot_grid(
  NULL,NULL,NULL,
  p_market+noleg, NULL, p_hml+noleg,
  p_wml+noleg, NULL, p_hml_wml+noleg,
  nrow = 3,
  rel_widths = c(1,0,1),
  rel_heights = c(.2,1,1)
)
p_grid<-p_grid+draw_grob(legend_b,2/3.3, -0.45, -0.2, 2.8)
p_grid

# save figure
ggsave(filename = "SR_CI.png",
       plot=p_grid,
       path = save_plots,
       width = 16, height = 10)




###########################################################
# COMPARISON IN SAMPLE AND OUT OF SAMPLE
###########################################################

setwd(path2)
files_out_s<-list.files(path = path2)

IS<-read.csv("is_1937-2019.csv")
OOS<-read.csv("OS_1937-2019.csv")

a<-mean(IS$VM_Mkt)/sd(IS$VM_Mkt)

b<-mean(IS$Mkt)/sd(IS$Mkt)
a-b

# perfrom sharpe ratio test
df_sr_out_s<-test_Sharpe_ratio(files=files_out_s,
                             original_portfolios=original_portfolios,
                             VM_portfolios=VM_portfolios,
                             ctr=control)

# get confidence intervals
out_CI<-CI_portfolio(df=df_sr_out_s,vars=VARS,CI=c(0.95))

# rename leverage to estimation
out_CI<-out_CI %>% 
  rename(
    estimation = leverage
  )





# rename portfolio names
port_names<-c("Market (Mkt-Rf)","Value (HML)",
              "Momentum (WML)","Value-Momentum (HML-WML)")
out_CI$portfolio<-rep(port_names,2)
#estimation_names<-c(rep(c('in sample'),4),rep(c('out of sample'),4))
#out_CI$estimation<-estimation_names
out_CI
  

# Plot parameters
XLIM<-c(-0.05,0.21) #monthly c(-0.05,0.21) #daily #monthly c(-0.01,0.05)
FONTSTYLE<-"Comic Sans MS"
FONTSIZE<-13
C1="#009E73" 
C2="#D55E00"
TITLE<-""

(1-pnorm(0.5525116))*2

stee<-0.008261506/0.5525116

0.008261506-qnorm(0.975)*stee
0.008261506+qnorm(0.975)*stee

p_compare<-ggplot(out_CI,aes(x = dsharpe, y = reorder(portfolio,dsharpe), xmin = conf.low, 
                 xmax = conf.high, color = estimation)) +
  geom_vline(xintercept = 0, linetype = 2, color = "gray50")+
  ggplot2::geom_errorbarh(position = ggstance::position_dodgev(height = .7), 
                          height = 0.4, size=1.25) +
  geom_point(position = ggstance::position_dodgev(height = .7), size = 3) +
  scale_x_continuous(
    limits = XLIM,
    #    expand = c(0, 0),
    name = "Difference in Sharpe ratio"
  ) +
  scale_y_discrete(name = NULL) +
  scale_color_manual(
    name = NULL,
    values = c(
      is = C1,
      os = C2),
    breaks = c("is", "os"),
    labels = c("95% CI in sample", "95% CI out of sample"),
    guide = guide_legend(direction = "vertical")
                         #title.position = "top",
                         #label.position = "bottom")
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal_hgrid() +
  theme_minimal_hgrid(13, rel_small = 1) +
  labs(title = TITLE)+
  theme(
    axis.line.x = element_line(color = "black"),
    axis.line.x.top = element_blank(),
    axis.ticks.x = element_line(color = "black"),
    axis.ticks.x.top = element_line(color = "gray50"),
    legend.key.height = grid::unit(8, "pt"),
    legend.key.width = grid::unit(30, "pt"),
    legend.spacing.x = grid::unit(2, "pt"),
    legend.spacing.y = grid::unit(3, "pt"),
    legend.box.background = element_rect(fill = "white", color = NA),
    legend.box.spacing = grid::unit(0, "pt"),
    legend.title.align = 0.1,
    legend.text=element_text(size=rel(1)),
    plot.title = element_text(color="black", 
                              size=FONTSIZE, 
                              face="bold.italic",
                              family=FONTSTYLE),
    axis.text=element_text(size=FONTSIZE,
                           family=FONTSTYLE),
    axis.title.x = element_text(size = FONTSIZE,
                                family=FONTSTYLE))


legend_cp <- get_legend(p_compare + theme(legend.position="bottom"))

noleg<-theme(legend.position="none")
p_grid<-plot_grid(
  p_compare+theme(legend.position="none"),
  nrow = 1,
  rel_heights = c(1)
)

p_compare_legend<-p_grid+draw_grob(legend_cp,2/2.9, -1.2, .4, 3)
p_compare_legend

#save figure
ggsave(filename = "insample_out_sample.png",
       plot=p_compare_legend,
       path = save_plots,
       width = 10, height = 5)







