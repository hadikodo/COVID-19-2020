package com.example.covid_19_detector;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import static com.example.covid_19_detector.AssetsUtils.loadLines;

public class MainActivity extends AppCompatActivity {
    public static final int RequestCode = 0;
    public static final String ALLOW_KEY = "ALLOWED";
    public static final String CAMERA_PREF = "camera_pref";
    private Bitmap photo;
    private Map<String, Float> floatMap;
    private ModelClassificator modelClassificator;
    private List<ClassificationResult> Results;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            modelClassificator = new ModelClassificator(this, new ModelConfig() {
                @Override
                public String getModelFilename() {
                    return "quantized_model.tflite";
                }
                @Override
                public String getLabelsFilename() {
                    return "labels.txt";
                }
                @Override
                public int getInputWidth() {
                    return 500;
                }
                @Override
                public int getInputHeight() {
                    return 500;
                }
                @Override
                public int getInputSize() {
                    return getInputWidth() * getInputHeight() * getChannelsCount() * FLOAT_BYTES_COUNT;
                }
                @Override
                public int getChannelsCount() {
                    return 1;
                }
                @Override
                public float getStd() {
                    return 128.0f;
                }
                @Override
                public float getMean() {
                    return 0.0f;
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void rtc_click(View view) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            if (getFromPref(this, ALLOW_KEY)) {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.CAMERA},
                        RequestCode);
            } else if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.CAMERA)) {
                    ActivityCompat.requestPermissions(MainActivity.this,
                            new String[]{Manifest.permission.CAMERA},
                            RequestCode);
                } else {
                    ActivityCompat.requestPermissions(this,
                            new String[]{Manifest.permission.CAMERA},
                            RequestCode);
                }
            }
        } else {
            Intent i = new Intent(MainActivity.this,CameraActivity.class);
            startActivity(i);
        }
    }

    public void iph_click(View view) {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, 1);
    }


    public static void saveToPreferences(MainActivity context, String key, Boolean allowed) {
        SharedPreferences myPrefs = context.getSharedPreferences(CAMERA_PREF,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor prefsEditor = myPrefs.edit();
        prefsEditor.putBoolean(key, allowed);
        prefsEditor.commit();
    }

    public static Boolean getFromPref(Context context, String key) {
        SharedPreferences myPrefs = context.getSharedPreferences(CAMERA_PREF,
                Context.MODE_PRIVATE);
        return (myPrefs.getBoolean(key, false));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case RequestCode: {
                for (int i = 0, len = permissions.length; i < len; i++) {
                    String permission = permissions[i];
                    if (grantResults[i] == PackageManager.PERMISSION_DENIED) {
                        boolean
                                showRationale =
                                ActivityCompat.shouldShowRequestPermissionRationale(
                                        this, permission);
                        if (showRationale) {
                            Log.v("Permission", "Permission Denied");
                        } else if (!showRationale) {
                            saveToPreferences(this, ALLOW_KEY, true);
                        }
                    } else {
                        Intent j = new Intent(MainActivity.this,CameraActivity.class);
                        startActivity(j);
                    }
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == Activity.RESULT_OK) {
            try {
                final Uri imageUri = data.getData();
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                    photo = selectedImage;
                    List<String> labels = null;
                    try {
                        labels = loadLines(this, "labels.txt");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
                        ByteArrayOutputStream stream = new ByteArrayOutputStream();
                        photo.compress(Bitmap.CompressFormat.PNG, 100, stream);
                        byte[] byteArray = stream.toByteArray();
                        Results = modelClassificator.process(photo);
                        Intent j = new Intent(this, ShowResultActivity.class);
                        j.putExtra("image", byteArray);
                        j.putExtra("label", ResultUtils.resultsToStr(Results));
                        startActivity(j);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

            } catch (FileNotFoundException e) {
                e.printStackTrace();
                Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
            }

        }
        else {
            Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
        }
    }
}
