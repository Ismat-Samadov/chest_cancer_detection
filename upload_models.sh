#!/bin/bash
flyctl ssh sftp shell << 'SFTP_COMMANDS'
cd /app/models
put models/chest_ct_binary_classifier.keras
put models/chest_ct_binary_classifier.h5
put models/chest_ct_binary_classifier.tflite
mkdir -p chest_ct_binary_classifier_tf/assets
mkdir -p chest_ct_binary_classifier_tf/variables
put models/chest_ct_binary_classifier_tf/assets/fingerprint.pb chest_ct_binary_classifier_tf/assets/
put models/chest_ct_binary_classifier_tf/assets/saved_model.pb chest_ct_binary_classifier_tf/assets/
put models/chest_ct_binary_classifier_tf/variables/variables.data-00000-of-00001 chest_ct_binary_classifier_tf/variables/
put models/chest_ct_binary_classifier_tf/variables/variables.index chest_ct_binary_classifier_tf/variables/
exit
SFTP_COMMANDS
