package com.example.nerv_io.data

import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@Parcelize
data class Hospital (
        val Nama: String ?="",
        val Alamat: String ?="",
        val Latitude: String ?="",
        val Longitude: String ?="",
        val Image: String?=""
        ) : Parcelable