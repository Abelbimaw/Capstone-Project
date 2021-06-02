package com.example.nerv_io.adapter.utils

import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@Parcelize
data class IsOpen (
    var isOpen : Boolean ? = false
) : Parcelable