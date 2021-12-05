import tensorflow as tf

inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
        )

# for layer in inception.layers:
    # print(layer)


print(inception.summary())

# image = tf.random(1, 299, 299, 3)

# out = inception(image)

print(inception.get_layer(name='mixed10').output.shape)