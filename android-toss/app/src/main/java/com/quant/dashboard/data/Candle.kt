package com.quant.dashboard.data

/** 봉 하나 — 시각(epoch 초) + 시고저종. 일봉·1분봉이 같은 모양을 쓴다. */
data class Candle(
    val t: Long,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
)
