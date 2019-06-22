test_Sharpe_ratio<-function(files=NULL,
                            original_portfolios=NULL,
                            VM_portfolios=NULL,
                            ctr=list(type = 1, 
                                     hac=TRUE,
                                     bBoot = 0)){
  
  # dataframe to rbind everything  
  df_sr_all<-NULL
  
  for (f in files){
    
    # get csv file
    out<-read.csv(f)
    
    # to rbind the portfolios
    
    for (i in c(1:length(VM_portfolios))){
      
      # get name of portfolios
      op_str<-original_portfolios[i]
      vm_str<-VM_portfolios[i]
      # get vectors
      op_df <-out%>% pull(op_str)
      vm_df <-out%>% pull(vm_str)
      ## Run Sharpe testing (asymptotic hac)
      c<-sharpeTesting(vm_df, op_df, control = ctr)
      out_SR<-data.frame(c$n,c$dsharpe,c$tstat,c$pval)
      
      colnames(out_SR)<-c("n","dsharpe","tstat","pval")
      # Compute standard error
      se<-out_SR$dsharpe/out_SR$tstat
      out_SR$se<-se
      
      # add period leverage and portfolio
      out_SR$period<-substr(f, 4, 12)
      out_SR$leverage<-substr(f, 1, 2)
      out_SR$portfolio<-op_str
      # add together
      df_sr_all<-rbind(df_sr_all,out_SR)
    }
    
  }
  return(df_sr_all)
}


CI_portfolio<-function(df=NULL, 
                       portfolio_n=NULL, 
                       leverage_n=NULL,
                       CI=c(0.99,0.95,0.90),
                       vars=NULL){
  
  ci_df<-NULL
  for (level in CI){
    df$level<-paste0(signif(100*level, 2), "%")
    df$conf.low<-df$dsharpe-df$se*qnorm(p=1-(1-level)/2)
    df$conf.high<-df$dsharpe+df$se*qnorm(p=1-(1-level)/2)
    ci_df<-rbind(ci_df, df)
  }
  ci_df<-select(ci_df,vars)
  return (ci_df)
}


plot_CI<-function(df=NULL, 
                  title=NULL, 
                  xlimits=c(-0.13, .24),
                  color1="#0072B2",
                  color2="black",
                  size_color2=2.5,
                  fontstyle="Comic Sans MS"){
  
  p<-ggplot(df, aes(x = dsharpe, y = reorder(portfolio,dsharpe))) +
    geom_vline(xintercept = 0, linetype = 2, color = "gray50") +
    geom_errorbarh(
      aes(xmin = conf.low, xmax = conf.high, 
          color = level,size=level),
      height = 0
    ) +
    geom_errorbarh(
      aes(xmin = conf.low, xmax = conf.high, color = level),
      height = 0
    ) +
    geom_point(data = filter(df, level == "90%"), 
               size = 3, color = color2) +
    scale_x_continuous(
      name = "Differences in Sharpe ratio",
      limits = xlimits
    ) +
    scale_y_discrete(
      name = NULL
    ) +
    scale_color_manual(
      name = "confidence level",
      values = c(
        `90%` = desaturate(darken(color1, .2), .3),
        `95%` = desaturate(lighten(color1, .2), .3),
        `99%` = desaturate(lighten(color1, .4), .3)
      ),
      guide = guide_legend(
        direction = "horizontal",
        title.position = "top",
        label.position = "bottom"
      )
    ) +
    scale_size_manual(
      name = "confidence level",
      values = c(
        `90%` = 3,
        `95%` = 1.5,
        `99%` = 0.75
      ),
      guide = guide_legend(
        direction = "horizontal",
        title.position = "top",
        label.position = "bottom"
      )
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal_hgrid() +
    theme_minimal_hgrid(13, rel_small = 1) +
    labs(title = title)+
    theme(
      axis.line.x = element_line(color = "black"),
      axis.line.x.top = element_blank(),
      axis.ticks.x = element_line(color = "black"),
      axis.ticks.x.top = element_line(color = "gray50"),
      #axis.title.x = element_text(hjust = 1),
      legend.justification = c(1, 0),
      legend.position = c(1, .01),
      legend.key.height = grid::unit(6, "pt"),
      legend.key.width = grid::unit(30, "pt"),
      legend.spacing.x = grid::unit(6, "pt"),
      legend.spacing.y = grid::unit(3, "pt"),
      legend.box.background = element_rect(fill = "white", color = NA),
      legend.box.spacing = grid::unit(0, "pt"),
      legend.title.align = 0.5,
      plot.title = element_text(color="black", 
                                size=14, 
                                face="bold.italic",
                                family=fontstyle),
      axis.text=element_text(size=10, 
                             family=fontstyle),
      axis.title.x = element_text(size = 10, 
                                  family=fontstyle)
      
    )
  return(p)
}


CI_plot_2<-function(df=NULL, 
                    xlims=c(-0.1,0.2),
                    title=NULL, 
                    col1="#009E73", 
                    col2="#D55E00",
                    fontstyle="Comic Sans MS",
                    fontsize=16) { 
  
  p<-ggplot(df,aes(x = dsharpe, y = period, xmin = conf.low, 
                   xmax = conf.high, color = leverage)) +
    geom_vline(xintercept = 0, linetype = 2, color = "gray50")+
    ggplot2::geom_errorbarh(position = ggstance::position_dodgev(height = .7), 
                            height = 0.5, size=1.25) +
    geom_point(position = ggstance::position_dodgev(height = .7), size = 3) +
    scale_x_continuous(
      limits = xlims,
      #    expand = c(0, 0),
      name = "Difference in Sharpe ratio"
    ) +
    scale_y_discrete(name = NULL) +
    scale_color_manual(
      name = NULL,
      values = c(
        ul = col1,
        ll = col2),
      breaks = c("ll", "ul"),
      labels = c("95% CI limited leverage", "95% CI unlimited leverage")
      #guide = guide_legend(direction = "horizontal")
    ) +
    coord_cartesian(clip = "off") +
    theme_minimal_hgrid() +
    theme_minimal_hgrid(15, rel_small = 1) +
    labs(title = title)+
    theme(
      axis.line.x = element_line(color = "black"),
      axis.line.x.top = element_blank(),
      axis.ticks.x = element_line(color = "black"),
      axis.ticks.x.top = element_line(color = "gray50"),
      legend.key.height = grid::unit(6, "pt"),
      legend.key.width = grid::unit(25, "pt"),
      legend.spacing.x = grid::unit(15, "pt"),
      legend.spacing.y = grid::unit(3, "pt"),
      legend.box.background = element_rect(fill = "white", color = NA),
      legend.box.spacing = grid::unit(0, "pt"),
      legend.title.align = 0.5,
      legend.text=element_text(size=rel(1)),
      plot.title = element_text(color="black", 
                                size=fontsize, 
                                face="bold.italic",
                                family=fontstyle),
      axis.text=element_text(size=fontsize,
                             family=fontstyle),
      axis.title.x = element_text(size = fontsize,
                                  family=fontstyle)
      
    )
  return(p)
} 






theme_cowplot <- function(font_size = 14, font_family = "", line_size = .5,
                          rel_small = 12/14, rel_tiny = 11/14, rel_large = 16/14) {
  half_line <- font_size / 2
  small_size <- rel_small * font_size
  
  # work off of theme_grey just in case some new theme element comes along
  theme_grey(base_size = font_size, base_family = font_family) %+replace%
    theme(
      line              = element_line(color = "black", size = line_size, linetype = 1, lineend = "butt"),
      rect              = element_rect(fill = NA, color = NA, size = line_size, linetype = 1),
      text              = element_text(family = font_family, face = "plain", color = "black",
                                       size = font_size, hjust = 0.5, vjust = 0.5, angle = 0, lineheight = .9,
                                       margin = margin(), debug = FALSE),
      
      axis.line         = element_line(color = "black", size = line_size, lineend = "square"),
      axis.line.x       = NULL,
      axis.line.y       = NULL,
      axis.text         = element_text(color = "black", size = small_size),
      axis.text.x       = element_text(margin = margin(t = small_size / 4), vjust = 1),
      axis.text.x.top   = element_text(margin = margin(b = small_size / 4), vjust = 0),
      axis.text.y       = element_text(margin = margin(r = small_size / 4), hjust = 1),
      axis.text.y.right = element_text(margin = margin(l = small_size / 4), hjust = 0),
      axis.ticks        = element_line(color = "black", size = line_size),
      axis.ticks.length = unit(half_line / 2, "pt"),
      axis.title.x      = element_text(
        margin = margin(t = half_line / 2),
        vjust = 1
      ),
      axis.title.x.top  = element_text(
        margin = margin(b = half_line / 2),
        vjust = 0
      ),
      axis.title.y      = element_text(
        angle = 90,
        margin = margin(r = half_line / 2),
        vjust = 1
      ),
      axis.title.y.right = element_text(
        angle = -90,
        margin = margin(l = half_line / 2),
        vjust = 0
      ),
      
      
      legend.background = element_blank(),
      legend.spacing    = unit(font_size, "pt"),
      legend.spacing.x  = NULL,
      legend.spacing.y  = NULL,
      legend.margin     = margin(0, 0, 0, 0),
      legend.key        = element_blank(),
      legend.key.size   = unit(1.1 * font_size, "pt"),
      legend.key.height = NULL,
      legend.key.width  = NULL,
      legend.text       = element_text(size = rel(rel_small)),
      legend.text.align = NULL,
      legend.title      = element_text(hjust = 0),
      legend.title.align = NULL,
      legend.position   = "right",
      legend.direction  = NULL,
      legend.justification = c("left", "center"),
      legend.box        = NULL,
      legend.box.margin =  margin(0, 0, 0, 0),
      legend.box.background = element_blank(),
      legend.box.spacing = unit(font_size, "pt"),
      
      panel.background  = element_blank(),
      panel.border      = element_blank(),
      panel.grid        = element_blank(),
      panel.grid.major  = NULL,
      panel.grid.minor  = NULL,
      panel.grid.major.x = NULL,
      panel.grid.major.y = NULL,
      panel.grid.minor.x = NULL,
      panel.grid.minor.y = NULL,
      panel.spacing     = unit(half_line, "pt"),
      panel.spacing.x   = NULL,
      panel.spacing.y   = NULL,
      panel.ontop       = FALSE,
      
      strip.background  = element_rect(fill = "grey80"),
      strip.text        = element_text(
        size = rel(rel_small),
        margin = margin(half_line / 2, half_line / 2,
                        half_line / 2, half_line / 2)
      ),
      strip.text.x      = NULL,
      strip.text.y      = element_text(angle = -90),
      strip.placement   = "inside",
      strip.placement.x =  NULL,
      strip.placement.y =  NULL,
      strip.switch.pad.grid = unit(half_line / 2, "pt"),
      strip.switch.pad.wrap = unit(half_line / 2, "pt"),
      
      plot.background   = element_blank(),
      plot.title        = element_text(
        face = "bold",
        size = rel(rel_large),
        hjust = 0, vjust = 1,
        margin = margin(b = half_line)
      ),
      plot.subtitle     = element_text(
        size = rel(rel_small),
        hjust = 0, vjust = 1,
        margin = margin(b = half_line)
      ),
      plot.caption      = element_text(
        size = rel(rel_tiny),
        hjust = 1, vjust = 1,
        margin = margin(t = half_line)
      ),
      plot.tag           = element_text(
        face = "bold",
        hjust = 0, vjust = 0.7
      ),
      plot.tag.position = c(0, 1),
      plot.margin       = margin(half_line, half_line, half_line, half_line),
      
      complete = TRUE
    )
}



theme_minimal_grid <- function(font_size = 14, font_family = "", line_size = .5,
                               rel_small = 12/14, rel_tiny = 11/14, rel_large = 16/14,
                               color = "grey85", colour) {
  if (!missing(colour)) {
    color <- colour
  }
  
  # Starts with theme_cowplot and then modifies some parts
  theme_cowplot(font_size = font_size, font_family = font_family, line_size = line_size,
                rel_small = rel_small, rel_tiny = rel_tiny, rel_large = rel_large) %+replace%
    theme(
      # make grid lines
      panel.grid        = element_line(color = color,
                                       size = line_size),
      panel.grid.minor  = element_blank(),
      
      # adjust axis tickmarks
      axis.ticks        = element_line(color = color, size = line_size),
      
      # no x or y axis lines
      axis.line.x       = element_blank(),
      axis.line.y       = element_blank(),
      
      # no filled background for facted plots
      strip.background = element_blank(),
      
      complete = TRUE
    )
}

#' @rdname theme_minimal_grid
#' @export
theme_minimal_hgrid <- function(font_size = 14, font_family = "", line_size = .5,
                                rel_small = 12/14, rel_tiny = 11/14, rel_large = 16/14,
                                color = "grey85", colour) {
  if (!missing(colour)) {
    color <- colour
  }
  
  # Starts with theme_grid and then modifies some parts
  theme_minimal_grid(font_size = font_size, font_family = font_family, line_size = line_size,
                     rel_small = rel_small, rel_tiny = rel_tiny, rel_large = rel_large,
                     color = color) %+replace%
    theme (
      # no vertical grid lines
      panel.grid.major.x = element_blank(),
      
      # add a x axis line
      axis.line.x       = element_line(color = color, size = line_size),
      
      complete = TRUE
    )
}


theme_dviz_hgrid <- function(font_size = 14, font_family = "") {
  color = "grey90"
  line_size = 0.5
  
  theme_cowplot(font_size = font_size, font_family = font_family) %+replace%
    theme(
      # make horizontal grid lines
      panel.grid.major = element_line(colour = color,
                                      size = line_size),
      panel.grid.major.x = element_blank(),
      
      # adjust axis tickmarks
      axis.ticks        = element_line(colour = color, size = line_size),
      
      # adjust x axis
      axis.line.x       = element_line(colour = color, size = line_size),
      # no y axis line
      axis.line.y       = element_blank()
    )
}