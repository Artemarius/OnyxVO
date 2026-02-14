package com.onyxvo.app

class NativeBridge {

    companion object {
        init {
            System.loadLibrary("onyx_vo")
        }
    }

    external fun nativeInit()
    external fun nativeGetVersion(): String
}
