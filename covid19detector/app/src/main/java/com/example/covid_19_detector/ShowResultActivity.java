package com.example.covid_19_detector;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class ShowResultActivity extends AppCompatActivity {
    private Bitmap photo;
    private ImageView imageview;
    private TextView textView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show_result);
        byte[] byteArray = getIntent().getByteArrayExtra("image");
        photo = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        imageview=findViewById(R.id.imageView);
        textView=findViewById(R.id.textView);
        imageview.setImageBitmap(photo);
        textView.setText(getIntent().getStringExtra("label"));
    }
}