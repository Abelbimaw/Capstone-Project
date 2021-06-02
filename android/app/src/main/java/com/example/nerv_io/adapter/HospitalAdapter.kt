package com.example.nerv_io.adapter

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
import com.example.nerv_io.data.Hospital
import kotlinx.android.synthetic.main.item_hospital.view.*

class HospitalAdapter(private val context: Context?, private val listFeed: MutableList<Hospital>) :
    RecyclerView.Adapter<HospitalAdapter.MyViewHolder>() {

    lateinit var glide: RequestManager

    interface OnItemClickListener {
        fun onItemClickListener(feed: Hospital)
    }

    inner class MyViewHolder(val view: View) : RecyclerView.ViewHolder(view) {
        fun bind(data: Hospital) {
            view.apply {
                Glide.with(context).load(data.Image)
                    .error(R.drawable.hospital)
                    .into(img_hospital)

                name_hospital.text = data.Nama
                address_hospital.text = data.Alamat

            }
        }

    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        val typeFollowing = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_hospital, parent, false)
        return MyViewHolder(typeFollowing)
    }

    override fun getItemCount(): Int = listFeed.size


    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        val feed = listFeed[position]
        glide = Glide.with(context!!)
        holder.bind(feed)
    }
}