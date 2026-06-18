package com.quant.dashboard

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.quant.dashboard.ui.AnalysisScreen
import com.quant.dashboard.ui.theme.QuantTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            QuantTheme {
                AnalysisScreen()
            }
        }
    }
}
