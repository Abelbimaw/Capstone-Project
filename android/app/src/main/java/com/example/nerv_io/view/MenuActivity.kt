package com.example.nerv_io.view

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import com.example.nerv_io.adapter.utils.Cons
import com.example.nerv_io.databinding.ActivityMenuBinding
import com.google.firebase.auth.FirebaseAuth
import com.orhanobut.hawk.Hawk

class MenuActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMenuBinding
    private lateinit var auth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(binding.root)
        Hawk.init(this).build()
        auth = FirebaseAuth.getInstance()

        binding.imgSignout.setOnClickListener {
            auth.signOut()
            Intent(this@MenuActivity, SignInActivity::class.java).also {
                it.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                startActivity(it)
            }
        }
        binding.btnTest.setOnClickListener {
            if (Hawk.get<String>(Cons.isOpen) == null) {
                showAlert()
            } else {
                val intent = Intent(this, DiagnosticTestActivity::class.java)
                startActivity(intent)
            }
        }
        binding.btnHistory.setOnClickListener {
            val intent = Intent(this, HistoryDiagnosisActivity::class.java)
            startActivity(intent)
        }
        binding.btnRecommen.setOnClickListener {
            val intent = Intent(this, HospitalRecomActivity::class.java)
            startActivity(intent)
        }

        binding.GuideBook.setOnClickListener{
            val intent = Intent(this, GuideBookActivity::class.java)
            startActivity(intent)
        }
    }

    override fun onResume() {
        super.onResume()
    }

    fun showAlert() {
        val builder = AlertDialog.Builder(this)
        //set title for alert dialog
        builder.setTitle("Information")
        //set message for alert dialog
        builder.setMessage("Please read guide test before you start")
        builder.setIcon(android.R.drawable.ic_dialog_alert)
        builder.setPositiveButton("OKE"){dialogInterface, which ->

        }
        val alertDialog: AlertDialog = builder.create()
        alertDialog.setCancelable(false)
        alertDialog.show()
    }

}