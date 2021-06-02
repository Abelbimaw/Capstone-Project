package com.example.nerv_io.view

import android.R
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.example.nerv_io.adapter.utils.Cons
import com.example.nerv_io.data.User
import com.example.nerv_io.databinding.ActivityDiagnosticTestBinding
import com.example.nerv_io.ml.ModelAkurasiOverfittingKecil
import com.google.firebase.firestore.FirebaseFirestore
import com.orhanobut.hawk.Hawk
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*


class DiagnosticTestActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDiagnosticTestBinding
    var db = FirebaseFirestore.getInstance()
    val profile by lazy { Hawk.get<User>(Cons.MyProfile) }


    private val item = arrayOf(
        "Male",
        "Female"
    )

    private val item1 = arrayOf(
        "True",
        "False"
    )
    private val item2 = arrayOf(
        "yes",
        "no"
    )
    private val item3 = arrayOf(
        "asymptomatic",
        "atypical angina",
        "non-anginal pain",
        "typical angina"
    )
    private val item4 = arrayOf(
        "normal",
        "having ST-T wave abnormality"
    )
    private val item5 = arrayOf(
        "upsloping",
        "flap",
        "downsloping"
    )
    private val item6 = arrayOf(
        "0",
        "1",
        "2",
        "3"
    )
    private val item7 = arrayOf(
        "normal",
        "fixed defect",
        "reversable Defect"
    )
    private var gender = 1F

    private var restBlood = 1F

    private var exerciseEngina = 1F

    private var chestPain = 1F

    private var restEletrco = 1F

    private var stSegment = 1F

    private var restMajorVessel = 1F

    private var restThalValue = 1F

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDiagnosticTestBinding.inflate(layoutInflater)
        setContentView(binding.root)
        Hawk.init(this).build()

        val actionbar = supportActionBar
        actionbar!!.title = "Diagnosis Test"
        actionbar.setDisplayHomeAsUpEnabled(true)
        actionbar.setDisplayHomeAsUpEnabled(true)


        spinnerAdapter()
        spinnerAdapterRestBlood()
        spinnerAdapterExerciseEngina()
        spinnerAdapterChestPain()
        spinnerAdapterRestingElectro()
        spinnerAdapterStSegment()
        spinnerAdapterMajorVessels()
        spinnerAdapterThalValue()
        binding.userName.setText(profile.name)

        val button: Button = binding.btnResult
        button.setOnClickListener{
            val userName: EditText = binding.userName
            val userAge: EditText = binding.userAge
            val userRestingBlood: EditText = binding.userRestBlood
            val userCholesterol: EditText = binding.userCholesterol
            val userHeartRate: EditText = binding.userHeartRate
            val userOldPeak: EditText = binding.userOldpeak


            if (userName.text.isEmpty()) {
                Toast.makeText(this, "Full Name is Required", Toast.LENGTH_SHORT).show()
            } else if (userAge.text.isEmpty()) {
                Toast.makeText(this, "Age is Required", Toast.LENGTH_SHORT).show()
            }else if (userRestingBlood.text.isEmpty()) {
                Toast.makeText(this, "Resting Blood Pressure is Required", Toast.LENGTH_SHORT).show()
            } else if (userCholesterol.text.isEmpty()) {
                Toast.makeText(this, "Cholesterol is Required", Toast.LENGTH_SHORT).show()
            } else if (userHeartRate.text.isEmpty()) {
                Toast.makeText(this, "Max heart Rate Achieved is Required", Toast.LENGTH_SHORT).show()
            } else if (userOldPeak.text.isEmpty()) {
                Toast.makeText(this, "Oldpeak is Required", Toast.LENGTH_SHORT).show()
            } else {

                val name = userName.text.toString()
                val age = userAge.text.toString().toFloat()
                val restingBlood = userRestingBlood.text.toString().toFloat()
                val cholesterol = userCholesterol.text.toString().toFloat()
                val heartRate = userHeartRate.text.toString().toFloat()
                val oldPeak = userOldPeak.text.toString().toFloat()

                val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(13 * 4)
                byteBuffer.putFloat(age)
                byteBuffer.putFloat(gender)
                byteBuffer.putFloat(chestPain)
                byteBuffer.putFloat(restingBlood)
                byteBuffer.putFloat(cholesterol)
                byteBuffer.putFloat(restBlood)
                byteBuffer.putFloat(restEletrco)
                byteBuffer.putFloat(heartRate)
                byteBuffer.putFloat(exerciseEngina)
                byteBuffer.putFloat(oldPeak)
                byteBuffer.putFloat(stSegment)
                byteBuffer.putFloat(restMajorVessel)
                byteBuffer.putFloat(restThalValue)


                val model = ModelAkurasiOverfittingKecil.newInstance(this)

                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 13), DataType.FLOAT32)
                inputFeature0.loadBuffer(byteBuffer)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

//            tv.text = "Heart Disease - " + outputFeature0[0].toString() + "\nNot Heart Disease - " + outputFeature0[1].toString()

                var newValueHeartDisease = outputFeature0[0].toString().substring(0..3)
                var newValueNotHeartDisease = outputFeature0[1].toString().substring(0..3)
                Log.e("VALUEEEEE", newValueHeartDisease)
                var toIntHeart = newValueHeartDisease.toFloat() * 100
                var toIntNotHeart = newValueNotHeartDisease.toFloat() * 100
                Log.e("VALUEEEEE222", toIntHeart.toInt().toString())
                val currentDate: String =
                    SimpleDateFormat("dd/MM/yyyy", Locale.getDefault()).format(Date())
                val city = hashMapOf(
                    "Age" to age,
                    "ChestPainType" to chestPain,
                    "Cholesterol" to cholesterol,
                    "ExerciseIncluded" to exerciseEngina,
                    "FastingBlood" to restBlood,
                    "FullName" to name,
                    "Gender" to gender.toString(),
                    "MaxHeartRate" to heartRate,
                    "NumberOfMajorVessels" to restMajorVessel,
                    "Oldpeak" to oldPeak,
                    "RestingBlood" to restingBlood,
                    "RestingElectroCardiography" to restEletrco,
                    "SlopeOfThePeakExercise" to stSegment,
                    "ThalValue" to restThalValue,
                    "UserID" to profile.userID,
                    "DateTime" to currentDate,
                    "Photo" to profile.photo,
                    "HeartDisease" to toIntHeart.toInt(),
                    "NotHeartDisease" to toIntNotHeart.toInt(),
                )


                db.collection("diagnostic").document(UUID.randomUUID().toString())
                    .set(city)
                    .addOnSuccessListener { Log.d("ss", "DocumentSnapshot successfully written!") }
                    .addOnFailureListener { e -> Log.w("ss", "Error writing document", e) }

                val intent = Intent(this, ResultDiagnosticActivity::class.java)
                intent.putExtra(ResultDiagnosticActivity.NAME, name)
                intent.putExtra(ResultDiagnosticActivity.AGE, age.toString())
                intent.putExtra(ResultDiagnosticActivity.Disease, toIntHeart.toInt().toString())
                startActivity(intent)

                model.close()
            }

        }

    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    private fun spinnerAdapter() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item)
        binding.spinnerGender.adapter = adapter
        binding.spinnerGender.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    gender = 1F
                } else {
                    gender = 0F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerGender.adapter = adapter
    }
    private fun spinnerAdapterRestBlood() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item1)
        binding.spinnerRestblood.adapter = adapter
        binding.spinnerRestblood.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 1) {
                    restBlood = 0F
                } else {
                    restBlood = 1F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerRestblood.adapter = adapter
    }
    private fun spinnerAdapterExerciseEngina() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item2)
        binding.spinnerExerciseEngina.adapter = adapter
        binding.spinnerExerciseEngina.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 1) {
                    exerciseEngina = 0F
                } else {
                    exerciseEngina = 1F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerExerciseEngina.adapter = adapter
    }
    private fun spinnerAdapterChestPain() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item3)
        binding.spinnerChestpain.adapter = adapter
        binding.spinnerChestpain.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    chestPain = 0F
                } else if (i == 1){
                    chestPain = 1F
                } else if (i == 2){
                    chestPain = 2F
                }else if (i == 3){
                    chestPain = 3F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerChestpain.adapter = adapter
    }
    private fun spinnerAdapterRestingElectro() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item4)
        binding.spinnerRestingElectro.adapter = adapter
        binding.spinnerRestingElectro.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    restEletrco = 1F
                } else {
                    restEletrco = 2F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerRestingElectro.adapter = adapter
    }
    private fun spinnerAdapterStSegment() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item5)
        binding.spinnerStsegment.adapter = adapter
        binding.spinnerStsegment.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    stSegment = 0F
                } else if(i == 1){
                    stSegment = 1F
                } else if(i == 2){
                    stSegment = 2F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerStsegment.adapter = adapter
    }
    private fun spinnerAdapterMajorVessels() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item6)
        binding.spinnerMajorvessels.adapter = adapter
        binding.spinnerMajorvessels.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    restMajorVessel = 0F
                } else if(i == 1){
                    restMajorVessel = 1F
                } else if(i == 2){
                    restMajorVessel = 2F
                } else if(i == 3){
                    restMajorVessel = 3F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerMajorvessels.adapter = adapter
    }
    private fun spinnerAdapterThalValue() {
        val adapter = ArrayAdapter(this, R.layout.simple_spinner_dropdown_item, item7)
        binding.spinnerThalvalue.adapter = adapter
        binding.spinnerThalvalue.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>, view: View, i: Int, l: Long) {
                if (i == 0) {
                    restThalValue = 1F
                } else if(i == 1){
                    restThalValue = 2F
                } else if(i == 2){
                    restThalValue = 3F
                }
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {}
        }
        binding.spinnerThalvalue.adapter = adapter
    }
}
