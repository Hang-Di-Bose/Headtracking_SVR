# HeadTracking
Predict future head location based on Bose AR rotation vector. Project is collaboration between AI and Data CED ML Satellite and CED ARCD.

## Installation
Download the repo locally and install the requirements

    ```
    git clone git@github.com:BoseCorp/HeadTracking.git
    cd HeadTracking
    pip install -r /path/to/requirements.txt
    ```

If you do not have a local copy of the RecordingSession repo, do the following: Note: if you already have a local copy of the repo, make sure your branch is up-to-date with the master branch of RecordingSession. Otherwise, stash your changes and pull the latest master branch.

    ```
    cd ..
    #clone the RecordingSession repo
    git clone git@github.com:BoseCorp/RecordingSession.git
    ```

## Run
Update your paths in [`train.sh`](train.sh)

Run `train.sh`

### [Link to wiki](https://wiki.bose.com/display/ADCML/Head+Tracking+using+IMU+and+Rotation+data)

### Notes
Development is in TF 1.15 because of [this open issue with TF 2.0](https://github.com/tensorflow/tensorflow/issues/34585). TF 1.15 is forwards compatible with TF 2.0 if you call `tf.compat.v1.enable_v2_behavior()`. 
