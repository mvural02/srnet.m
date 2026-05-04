import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

# Parametreler
data_dir = r"D:\yeni_D\elpv_edited_100_2\data"
batch_size = 128
img_height = 100
img_width = 100
epochs = 500
learning_rate = 0.001

# Veri setini yukleme ve %90 Egitim, %10 Dogrulama olarak ayirma
print("Veri seti yukleniyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
        validation_split=0.1,
            subset="training",
                seed=123,
                    image_size=(img_height, img_width),
                        batch_size=batch_size,
                            color_mode='grayscale' # imageSize = [100 100 1]
                            )

                            val_ds = tf.keras.utils.image_dataset_from_directory(
                                data_dir,
                                    validation_split=0.1,
                                        subset="validation",
                                            seed=123,
                                                image_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                        color_mode='grayscale'
                                                        )

                                                        # Sinif isimlerini al
                                                        class_names = train_ds.class_names
                                                        print(f"Bulunan siniflar: {class_names}")

                                                        # ResNet Modeli (MATLAB'daki resnetLayers(..., StackDepth=[3 4 23 3]) ResNet-101'e karsilik gelir)
                                                        # weights=None ile sifirdan egitiyoruz, input_shape=(100, 100, 1)
                                                        input_tensor = Input(shape=(img_height, img_width, 1))

                                                        base_model = ResNet101(
                                                            include_top=False, 
                                                                weights=None, 
                                                                    input_tensor=input_tensor,
                                                                        pooling='avg'
                                                                        )

                                                                        # Siniflandirma katmani
                                                                        x = base_model.output
                                                                        # numClasses = 2
                                                                        predictions = Dense(len(class_names), activation='softmax')(x)

                                                                        model = Model(inputs=base_model.input, outputs=predictions)

                                                                        # Optimizasyon ve Derleme
                                                                        model.compile(
                                                                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                                                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                                                                    metrics=['accuracy']
                                                                                    )

                                                                                    # Model Ozeti
                                                                                    model.summary()

                                                                                    # Model Egitimi
                                                                                    print("Egitim basliyor...")
                                                                                    history = model.fit(
                                                                                        train_ds,
                                                                                            validation_data=val_ds,
                                                                                                epochs=epochs,
                                                                                                    shuffle=True
                                                                                                    )
                                                                                                    
                                                                                                    # Modeli kaydet
                                                                                                    model.save('resnet_model.h5')
                                                                                                    print("Egitim tamamlandi ve model kaydedildi.")
