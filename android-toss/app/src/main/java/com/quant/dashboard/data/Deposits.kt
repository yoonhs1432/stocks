package com.quant.dashboard.data

/** 입금 한 건. */
data class Deposit(val date: String, val krw: Double)

/**
 * 원금(입금 누적) — **PC 서버에 있다.**
 *
 * 토스 API 에 입출금 내역이 없어 어딘가에 적어 둬야 하는 값이다. PC 에 두면 폰을 바꿔도
 * 남고, 웹으로 봐도 같은 값이 나온다. 여기서는 받아 둔 목록을 읽기만 한다
 * (`Store.syncFromServer()` 가 채운다).
 */
object Deposits {

    @Volatile private var list: List<Deposit> = emptyList()
    @Volatile private var total = 0.0

    internal fun set(items: List<Deposit>, sum: Double) {
        list = items
        total = sum
    }

    fun load(): List<Deposit> = list

    fun total(): Double = total

    /** 서버에 넣고 목록을 다시 받아 둔다. IO 디스패처에서 호출할 것. */
    fun add(date: String, krw: Double) {
        Server.addDeposit(date, krw)
        Store.syncFromServer(force = true)
    }

    fun removeAt(index: Int) {
        Server.removeDeposit(index)
        Store.syncFromServer(force = true)
    }
}
