package com.example.nerv_io.data

import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@Parcelize
data class Diagnostic (
    val Age: Int ?=0,
    val ChestPainType: Int ?=0,
    val Cholesterol: Int ?=0,
    val ExerciseIncluded: Int ?=0,
    val FastingBlood: Int?=0,
    val FullName: String?="",
    val Gender: String?="",
    val HeartDisease: Int?=0,
    val MaxHeartRate: Int?=0,
    val NotHeartDisease: Int?=0,
    val NumberOfMajorVessels: Float?=0F,
    val Oldpeak: Int?=0,
    val DateTime: String?="",
    val RestingBlood: Float?=0F,
    val RestingElectroCardiography: Float?=0F,
    val SlopeOfThePeakExercise: Float?=0F,
    val ThalValue: Float?=0F,
    val UserID: String?="",
    val Photo: String?="",
) : Parcelable
