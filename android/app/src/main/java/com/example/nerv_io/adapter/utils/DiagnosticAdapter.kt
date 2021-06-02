package com.example.nerv_io.adapter.utils

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.bumptech.glide.RequestManager
import com.example.nerv_io.R
import com.example.nerv_io.data.Diagnostic
import com.example.nerv_io.data.Hospital
import kotlinx.android.synthetic.main.item_history.view.*
import kotlinx.android.synthetic.main.item_hospital.view.*

class DiagnosticAdapter(private val context: Context?, private val listFeed: MutableList<Diagnostic>) :
    RecyclerView.Adapter<DiagnosticAdapter.MyViewHolder>() {

    lateinit var glide: RequestManager

    interface OnItemClickListener {
        fun onItemClickListener(feed: Diagnostic)
    }

    inner class MyViewHolder(val view: View) : RecyclerView.ViewHolder(view) {
        fun bind(data: Diagnostic) {
            view.apply {
                Glide.with(context).load(data.Photo)
                    .error(R.drawable.hospital)
                    .into(img_user)

                name_user.text = data.FullName
                date_user.text = data.DateTime
                tv_heart_disease.text = "Possibly declared cancer "+data.HeartDisease.toString().plus(" %")

            }
        }

    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        val typeFollowing = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return MyViewHolder(typeFollowing)
    }

    override fun getItemCount(): Int = listFeed.size


    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        val feed = listFeed[position]
        glide = Glide.with(context!!)
        holder.bind(feed)
    }
}