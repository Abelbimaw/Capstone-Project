package com.example.nerv_io.view

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.nerv_io.databinding.ActivityResultDiagnosticBinding


class ResultDiagnosticActivity : AppCompatActivity() {
    private lateinit var binding : ActivityResultDiagnosticBinding

    companion object{
        const val NAME = "NAME"
        const val AGE = "AGE"
        const val Disease = "Disease"
    }
    private lateinit var name: String
    private lateinit var age: String
    private var ValueDisease: String? = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultDiagnosticBinding.inflate(layoutInflater)
        setContentView(binding.root)


        name = intent.getStringExtra(NAME).toString()
        age = intent.getStringExtra(AGE).toString()
        ValueDisease = intent.getStringExtra(Disease)

        binding.valueResult.text = "The results of the diagnostic by the system showed: ${name} (${age} years)"
        binding.valueHeart.text = ValueDisease+" %"


        binding.backDashboard.setOnClickListener {
            val intent = Intent(this, MenuActivity::class.java)
            startActivity(intent)
        }

    }

}