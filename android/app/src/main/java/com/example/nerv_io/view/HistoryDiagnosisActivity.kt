package com.example.nerv_io.view

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.nerv_io.adapter.utils.Cons
import com.example.nerv_io.adapter.utils.DiagnosticAdapter
import com.example.nerv_io.data.Diagnostic
import com.example.nerv_io.data.User
import com.example.nerv_io.databinding.ActivityHistoryDiagnosticBinding
import com.google.firebase.firestore.FirebaseFirestore
import com.orhanobut.hawk.Hawk

class HistoryDiagnosisActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryDiagnosticBinding
    var db = FirebaseFirestore.getInstance()
    private var diagnostic: MutableList<Diagnostic> = mutableListOf()
    var adapter: DiagnosticAdapter? = null
    val profile: User by lazy { Hawk.get<User>(Cons.MyProfile) }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryDiagnosticBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val actionbar = supportActionBar
        actionbar!!.title = "Diagnosis History"
        actionbar.setDisplayHomeAsUpEnabled(true)
        actionbar.setDisplayHomeAsUpEnabled(true)

        Hawk.init(this).build()
        initData()
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    private fun initData() {
        db.collection("diagnostic").whereEqualTo("FullName", profile.name)
            .get().addOnCompleteListener {
                for (document in it.result!!) {
                    val feeds = document.toObject(Diagnostic::class.java)
                    diagnostic.add(feeds)
                }
                adapter = DiagnosticAdapter(this, diagnostic)
                binding.rvDiagnostic.also {
                    it.layoutManager = LinearLayoutManager(this)
                    it.setHasFixedSize(true)
                    it.adapter = adapter
                }

                    binding.progressBar.visibility = View.GONE
            }
    }
}