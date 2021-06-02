package com.example.nerv_io.view

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.nerv_io.R
import com.example.nerv_io.adapter.utils.Cons
import com.example.nerv_io.adapter.utils.IsOpen
import com.orhanobut.hawk.Hawk

class GuideBookActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guide_book)
        Hawk.init(this).build()
 
        Hawk.put(Cons.isOpen, IsOpen(isOpen = true))

        val actionbar = supportActionBar
        actionbar!!.title = "Guide Test"
        actionbar.setDisplayHomeAsUpEnabled(true)
        actionbar.setDisplayHomeAsUpEnabled(true)
    }
    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
}