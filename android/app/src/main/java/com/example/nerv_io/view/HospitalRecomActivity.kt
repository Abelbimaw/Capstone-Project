package com.example.nerv_io.view

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.nerv_io.adapter.HospitalAdapter
import com.example.nerv_io.data.Hospital
import com.example.nerv_io.databinding.ActivityHospitalRecomBinding
import com.google.firebase.firestore.FirebaseFirestore

class HospitalRecomActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHospitalRecomBinding
    var db = FirebaseFirestore.getInstance()
    private var listHospital: MutableList<Hospital> = mutableListOf()
    var adapter: HospitalAdapter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHospitalRecomBinding.inflate(layoutInflater)
        setContentView(binding.root)
        initData()

        val actionbar = supportActionBar
        actionbar!!.title = "Hospital Recommendation"
        actionbar.setDisplayHomeAsUpEnabled(true)
        actionbar.setDisplayHomeAsUpEnabled(true)
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    private fun initData() {
        db.collection("rs")
            .get().addOnCompleteListener {
                for (document in it.result!!) {
                    val feeds = document.toObject(Hospital::class.java)
                    listHospital.add(feeds)
                }
                adapter = HospitalAdapter(this, listHospital)
                binding.rvHospital.also {
                    it.layoutManager = LinearLayoutManager(this)
                    it.setHasFixedSize(true)
                    it.adapter = adapter
                }
                    binding.progressBar.visibility = View.GONE
            }
    }

}