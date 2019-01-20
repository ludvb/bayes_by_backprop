#!/usr/bin/env Rscript

library(ggpubr)
library(grid)
library(gridExtra)
library(gtable)
library(kableExtra)
library(tidyverse)
library(zeallot)


set.seed(12345)


enumerate <- function(xs) (
    transpose(list(1:length(xs), xs))
    %>% map(lift(function(i, x) c(list(i), as.list(x))))
)

`%?<-%` <- function(x, value) {
    if (!all(zeallot:::tree(substitute(x))
             %>% map_lgl(~exists(as.character(.)))))
    {
        zeallot:::multi_assign(substitute(x), value, env = parent.frame())
    }
}


# | -------------------


message("loading training data")
c(tdVi, tdMl) %?<-% (
    c("../training_data_var.csv", "../training_data_mle.csv")
    %>% map(function(f)
        read.csv(f, header=T)
        %>% as_tibble()
        %>% filter(epoch <= 10000)
        %>% mutate(validation = as.logical(validation))
        )
)

savePlot <- function(
    plot
  , name
  , width=unit(7, "in")
  , height=unit(3.8, "in")
  , fp = sprintf("/tmp/%s.svg", name)
  , ...
) {
    message(sprintf("saving %s...", fp))
    ggsave(fp, plot, width=width, height=height, ...)
    system(sprintf("sed -i 's/font-family: Arial;//' %s", fp))
}

plotTheme <- (
    theme_bw(base_family =  "Helvetica"
             )
    + theme_classic()
    + theme_minimal()
    + theme(
          legend.title=element_blank()
         ,panel.grid.major = element_blank()
         ,panel.grid.minor = element_blank()
      )
)


# | -------------------
# | Training data plots


lossPlot <- function(df) (
    ggplot(
        df
        %>% filter(type == "loss", validation == 0)
        %>% group_by(epoch, type)
        %>% summarize(loss=mean(value))
        %>% ungroup()
    )
    + aes(x=epoch, y=log(loss))
    + geom_line()
    + plotTheme
)

accuracyPlot <- function(df)  (
    ggplot(
        df
        %>% filter(type == "acc" | type == "wt_acc", validation == 1)
        %>% group_by(epoch, type)
        %>% summarize(value=mean(value))
        %>% ungroup()
        %>% mutate(type=fct_recode(type
                                  ,Accuracy = "acc"
                                  ,c("Wt. accuracy" = "wt_acc")
                                    ))
    )
    + aes(x=epoch, y=value, color=type)
    + geom_line()
    + plotTheme
    + ylim(0.4, 1.0)
)

c(lvi, lml) %<-% map(list(tdVi, tdMl), lossPlot)
c(avi, aml) %<-% map(list(tdVi, tdMl), accuracyPlot)
aLegend <- get_legend(avi)
c(lvi, lml, avi, aml) %<-% map(list(lvi, lml, avi, aml), ~ . + rremove("legend"))
trainingGrob <- arrangeGrob(
    gtable_cbind(
        gtable_rbind(ggplotGrob(lvi + ggtitle("LSTM, Bayesian") + rremove("x.title")), ggplotGrob(avi + rremove("y.title")))
       ,gtable_rbind(ggplotGrob(lml + ggtitle("LSTM, MLE") + rremove("x.title") + rremove("y.title")), ggplotGrob(aml + rremove("y.title")))
    )
   ,arrangeGrob(rectGrob(gp=gpar(col=NA)), aLegend, heights=c(0.5, 0.5))
   ,widths=c(0.8, 0.2)
)
savePlot(name="training_stats", trainingGrob)


# | -----------------


message("loading test data")
c(testVi, testMl) %?<-% (
    c("../apply-var/data.csv", "../apply-mle/data.csv")
    %>% map(function(f)
        read.csv(f, header=T)
        )
)
metadata <- (
    read.table("../test.tsv.gz", header=T)
    %>% as_tibble()
    %>% mutate(signal = as.logical(signal))
)

c(predVi, predMl) %?<-% (
    map(list(testVi, testMl), function(df)
        df
        %>% group_by(entry, sample)
        %>% arrange(position)
        %>% summarize(p=last(prediction1))
        %>% group_by(entry)
        %>% mutate(prediction=sum(p > 0.5) > n() / 2)
        %>% ungroup()
        %>% inner_join(metadata %>% select(entry, signal), by="entry")
    )
)

c(pmVi, pmMl) %?<-% (
    map(list(predVi, predMl), function(df)
        df
        %>% group_by(entry, prediction, signal)
        %>% summarize(p = median(p))
        %>% ungroup()
    )
)


# | -----------------
# | ROC plots


rocifydf <- function(df) {
    df <- (
        df
        %>% arrange(-p)
        %>% mutate(
                TPR = cumsum(signal) / sum(signal)
              , FPR = cumsum(!signal) / sum(!signal)
            )
    )
    auc <- sum((head(df$TPR, -1) + tail(df$TPR, -1)) / 2 * diff(df$FPR))
    list(df=df ,auc=auc)
}

c(c(rocdfVi, aucVi), c(rocdfMl, aucMl)) %<-% map(list(pmVi, pmMl), rocifydf)
rocdfVi$network = sprintf("LSTM, Bayesian (AUC: %.2f)", aucVi)
rocdfMl$network = sprintf("LSTM, MLE (AUC: %.2f)", aucMl)
rocPlot <- (
    ggplot(rbind(rocdfVi, rocdfMl))
    + aes(FPR, TPR, color=network)
    + geom_line()
    + geom_abline(slope=1, intercept=0, linetype=2, color="grey")
    + plotTheme
)
savePlot(name="roc", rocPlot, width=unit(5, "in"), height=unit(3, "in"))


accuracyTable <- function(df) (
    df
    %>% group_by(entry, prediction, signal)
    %>% summarize()
    %>% ungroup()
    %>% summarize(
            "Accuracy (%)"
            = 100 * sum(prediction == signal) / nrow(.)
          , "Weighted accuracy (%)"
            = 100 * (
                sum(prediction  &  signal) * 0.5 / sum( signal)
              + sum(!prediction & !signal) * 0.5 / sum(!signal)
            )
        )
)

(
    list(list("LSTM, Bayesian", predVi)
       , list("LSTM, MLE", predMl))
    %>% map(lift(function(name, df)
        accuracyTable(df)
        %>% do(bind_cols(data.frame(Model = name, stringsAsFactors = F), .))
        ))
    %>% invoke(bind_rows, .)
    %>% kable("pandoc", digits = 2)
    %>% write("/tmp/accuracy_table.md")
)

# | -----------------
# | Confusion plot


c(c(cvi1, cvi2), c(cml1, cml2)) %<-% (
    map(list(predVi, predMl), function(df)
        map(list(list(quo(..density..), "density")
               , list(quo(..scaled..), "scaled density"))
          , lift(function(y, ytitle)
              ggplot(
                  df %>% mutate(
                             class = map(
                                 list(signal, prediction)
                               , ~ifelse(., "T", "F")
                             ) %>% invoke(paste, ., sep = "/")
                         )
              )
              + aes(
                    p
                  , !! y
                  , color = class
                  , fill  = class
                )
              + geom_density(alpha=0.2)
              + xlim(0, 1)
              + xlab("ŷ")
              + ylab(ytitle)
              + plotTheme
              + theme(legend.title=element_text())
              + guides(
                    fill  = guide_legend(title="class/prediction")
                  , color = guide_legend(title="class/prediction"))
              )
            )
        )
)
legend <- get_legend(cvi1)
c(cvi1, cvi2, cml1, cml2) %<-% map(list(cvi1, cvi2, cml1, cml2), ~ . + rremove("legend"))
confusionGrob <- arrangeGrob(
    gtable_cbind(
        gtable_rbind(ggplotGrob(cvi1 + ggtitle("LSTM, Bayesian") + rremove("x.title")), ggplotGrob(cvi2))
       ,gtable_rbind(ggplotGrob(cml1 + ggtitle("LSTM, MLE") + rremove("x.title") + rremove("y.title")), ggplotGrob(cml2 + rremove("y.title")))
    )
   ,legend
   ,widths=c(0.7, 0.3)
)
savePlot(name="confusion", confusionGrob)


confusionMatrix <- function(df) (
    df
    %>% group_by(entry, prediction, signal)
    %>% summarize()
    %>% group_by(prediction, signal)
    %>% summarize(n=n())
    %>% spread(prediction, n)
)
c(cmVi, cmMl) %<-% map(list(predVi, predMl), confusionMatrix)
(
    inner_join(cmVi, cmMl, by="signal")
    %>% mutate(signal = ifelse(signal, "T", "F"))
    %>% bind_rows(bind_cols(data.frame(signal="Total", stringsAsFactors=F), summarize_at(., vars(-signal), sum)))
    %>% do(bind_cols(
            .
          , (.)
            %>% select(-signal)
            %>% compose(as_tibble, t)()
            %>% summarize_all(function(...) sum(...) / 2)
            %>% compose(as_tibble, t)()
            %>% `colnames<-`("Total")
        ))
    %>% do(`colnames<-`(., ifelse(grepl("^FALSE",    colnames(.)), "F", colnames(.))))
    %>% do(`colnames<-`(., ifelse(grepl("^TRUE",     colnames(.)), "T", colnames(.))))
    %>% do(`colnames<-`(., ifelse(grepl("^signal$",  colnames(.)), "",  colnames(.))))
    %>% kable()
    %>% add_header_above(c("", "Bayesian" = 2, "non-Bayesian" = 2, ""))
    %>% write("/tmp/confusion.html")
)


# | -----------------
# | False positives


set.seed(17742)
falsePositives <- (
    predVi
    %>% filter(prediction == 1, signal == 0)
    %>% pull(entry)
    %>% unique()
    %>% as.character()
    %>% sample(6)
    %>% imap(function(e, i)
        ggplot(predVi %>% filter(entry==e))
        + aes(p)
        + geom_density()
        + geom_rug()
        + geom_vline(
              xintercept = first(
                  predMl %>% filter(entry==e) %>% pull(p)
              )
            , linetype=2
            , color="red"
          )
        + xlab("ŷ")
        + xlim(0, 1)
        + ggtitle(e)
        + plotTheme
        + (if (i %% 3 != 1) rremove("y.title") else NULL)
        )
    %>% invoke(grid.arrange, ., ncol=3)
)
savePlot(name="falsepositive", falsePositives)


# | -----------------
# | Tracked output


summarized <- (
    testVi
    %>% group_by(entry, position)
    %>% summarize(median = median(prediction1)
                 ,min = quantile(prediction1, 0.05)
                 ,max = quantile(prediction1, 0.95))
    %>% group_by(entry)
    %>% mutate(diff = max(last(median) / first(median), first(median) / last(median)))
    %>% arrange(-diff)
)

confusionClass <- (
    predVi %>%
    group_by(entry, prediction, signal) %>%
    summarize()
)

set.seed(33030)
trackedOutput <- (
    as.matrix(expand.grid(c(T, F), c(T, F)))
    %>% array_branch(1)
    %>% enumerate()
    %>% map(lift(function(i, s, p)
        confusionClass
        %>% filter(signal == s, prediction == p)
        %>% pull(entry)
        %>% sample(3)
        %>% map(function(e)
            ggplot(summarized %>% filter(entry==e))
            + aes(x=position, y=median, ymin=min, ymax=max, fill="band")
            + geom_ribbon()
            + geom_line()
            + plotTheme
            + guides(fill=F)
            + ylab("ŷ")
            + ggtitle(e)
            + (if (i != 4) rremove("x.title") else NULL)
            )
        %>% invoke(
                function(p1, ...) (
                    do.call(
                        gtable_cbind
                      , c(list(p1)
                        , list(...) %>% map(~. + rremove("y.title")))
                        %>% map(ggplotGrob))
                    %>% gtable_add_cols(unit(0.1, "npc"))
                    %>% gtable_add_grob(
                            t=7
                          , l=28
                          , textGrob(do.call(sprintf, c(list("%s/%s"), ifelse(c(s, p), "T", "F"))))
                        )
                )
               ,.
            )
    ))
    %>% invoke(gtable_rbind, .)
)
savePlot(name="trackedoutput", trackedOutput, height=unit(5, "in"))
