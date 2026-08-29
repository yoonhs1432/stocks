package com.quant.dashboard

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.Store
import com.quant.dashboard.ui.AppScaffold
import com.quant.dashboard.ui.theme.QuantTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Store.init(applicationContext)
        BrokerCreds.init(applicationContext)
        setContent {
            QuantTheme {
                AppScaffold()
            }
        }
    }
}
