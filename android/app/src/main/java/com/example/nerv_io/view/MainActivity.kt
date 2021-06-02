package com.example.nerv_io.view

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import com.example.nerv_io.R
import com.example.nerv_io.adapter.utils.Cons
import com.example.nerv_io.data.User
import com.google.firebase.auth.FirebaseAuth
import com.orhanobut.hawk.Hawk


class MainActivity : AppCompatActivity() {
    private lateinit var firebaseAuth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        firebaseAuth = FirebaseAuth.getInstance()
        val user = firebaseAuth.currentUser

        Handler().postDelayed({
            if (user != null){
                val menuIntent = Intent(this, MenuActivity::class.java)
                startActivity(menuIntent)
            }else{
                val logInIntent = Intent(this, SignInActivity::class.java)
                startActivity(logInIntent)
            }
        },2000)
    }
}
