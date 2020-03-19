package com.example.android.pytorch_model_app;

import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.support.annotation.RequiresApi;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.Environment;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import org.json.*;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Utility class to change images.
 **/
class ImageUtils {
    /**
     * Returns a transformation matrix from one reference frame into another.
     * Handles cropping (if maintaining aspect ratio is desired) and rotation.
     *
     * @param srcWidth Width of source frame.
     * @param srcHeight Height of source frame.
     * @param dstWidth Width of destination frame.
     * @param dstHeight Height of destination frame.
     * @param applyRotation Amount of rotation to apply from one frame to another.
     *  Must be a multiple of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
     * cropping the image if necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }


    public static Bitmap processBitmap(Bitmap source,int size){

        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);

        Matrix frameToCropTransformations = getTransformationMatrix(image_width,image_height,size,size,0,false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(source, frameToCropTransformations, null);

        return croppedBitmap;


    }

    public static float[] normalizeBitmap(Bitmap source,int size,float mean,float std){

        float[] output = new float[size * size * 3];

        int[] intValues = new int[source.getHeight() * source.getWidth()];

        source.getPixels(intValues, 0, source.getWidth(), 0, 0, source.getWidth(), source.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            output[i * 3] = (((val >> 16) & 0xFF) - mean)/std;
            output[i * 3 + 1] = (((val >> 8) & 0xFF) - mean)/std;
            output[i * 3 + 2] = ((val & 0xFF) - mean)/std;
        }

        return output;

    }

    public static Object[] argmax(float[] array){


        int best = -1;
        float best_confidence = 0.0f;

        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};

    }


    public static String getLabel( InputStream jsonStream,int index){
        String label = "";
        try {

            byte[] jsonData = new byte[jsonStream.available()];
            jsonStream.read(jsonData);
            jsonStream.close();

            String jsonString = new String(jsonData,"utf-8");

            JSONObject object = new JSONObject(jsonString);

            label = object.getString(String.valueOf(index));



        }
        catch (Exception e){


        }
        return label;
    }
}


public class MainActivity extends AppCompatActivity {

    //Loading the Tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/squeezenet.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224, 224, 3};

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;

    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public Object[] argmax(float[] array) {


        int best = -1;
        float best_confidence = 0.0f;

        for (int i = 0; i < array.length; i++) {

            float value = array[i];

            if (value > best_confidence) {

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best, best_confidence};


    }

    public void predict(final Bitmap bitmap) {


        //Runs inference in background thread
        new AsyncTask<Integer, Integer, Integer>() {

            @Override

            protected Integer doInBackground(Integer... params) {

                //Resize the image into 224 x 224
                Bitmap resized_image = ImageUtils.processBitmap(bitmap, 224);

                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image, 224, 127.5f, 1.0f);

                //Pass input into the tensorflow
                tf.feed(INPUT_NAME, floatValues, 1, 224, 224, 3);

                //compute predictions
                tf.run(new String[]{OUTPUT_NAME});

                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME, PREDICTIONS);

                //Obtained highest prediction
                Object[] results = argmax(PREDICTIONS);


                int class_index = (Integer) results[0];
                float confidence = (Float) results[1];


                try {

                    final String conf = String.valueOf(confidence * 100).substring(0, 5);

                    //Convert predicted class index into actual label name
                    final String label = ImageUtils.getLabel(getAssets().open("labels.json"), class_index);


                    //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {

                            progressBar.dismiss();
                            resultView.setText(label + " : " + conf + "%");

                        }
                    });
                } catch (Exception e) {
                }
                return 0;
            }


        }.execute(0);

    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Toolbar toolbar = (Toolbar) findViewById(R.id.app_bar);
        //setSupportActionBar(toolbar);

        //initialize tensorflow with the AssetManager and the Model
        tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);

        imageView = (ImageView) findViewById(R.id.imageview);
        resultView = (TextView) findViewById(R.id.results);

        progressBar = Snackbar.make(imageView, "PROCESSING IMAGE", Snackbar.LENGTH_INDEFINITE);


        final FloatingActionButton predict = (FloatingActionButton) findViewById(R.id.predict);
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {


                try {

                    //READ THE IMAGE FROM ASSETS FOLDER
               InputStream imageStream = getAssets().open("broc.jif");

                    Bitmap bitmap = BitmapFactory.decodeStream(imageStream);

                    imageView.setImageBitmap(bitmap);

                    progressBar.show();

                    predict(bitmap);
                } catch (Exception e) {

                }

            }
        });
    }
}