-- ============================================================
-- stock_forecast_daily
-- predict.py 每次執行後寫入未來 30 天逐日預測軌跡
-- PRIMARY KEY: (predict_date, stock_id, forecast_date)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.stock_forecast_daily (
    predict_date       date          NOT NULL,
    stock_id           varchar(50)   NOT NULL,
    forecast_date      date          NOT NULL,
    day_offset         int           NOT NULL,
    price_q10          numeric(20,4) NULL,
    price_q25          numeric(20,4) NULL,
    price_q50          numeric(20,4) NULL,
    price_q75          numeric(20,4) NULL,
    price_q90          numeric(20,4) NULL,
    ensemble_price     numeric(20,4) NULL,
    current_close      numeric(20,4) NULL,
    prob_up            numeric(5,4)  NULL,
    confidence_level   varchar(50)   NULL,
    model_agreement    numeric(5,4)  NULL,
    xgb_prob           numeric(5,4)  NULL,
    lgb_prob           numeric(5,4)  NULL,
    tft_prob           numeric(5,4)  NULL,
    extreme_valuation  boolean       DEFAULT false,
    macro_shock        boolean       DEFAULT false,
    created_at         timestamptz   DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT stock_forecast_daily_pkey
        PRIMARY KEY (predict_date, stock_id, forecast_date)
);

CREATE INDEX IF NOT EXISTS idx_sfd_stock_predict
    ON public.stock_forecast_daily USING btree (stock_id, predict_date DESC);

CREATE INDEX IF NOT EXISTS idx_sfd_forecast_date
    ON public.stock_forecast_daily USING btree (forecast_date);
