# back up data, checkpoints, and logs to s3

echo "backing up data to s3"
FOLDERS=("clip_data")
for folder in ${FOLDERS[@]}; do
    SRC="./.cache/$folder/"
    DEST="s3://$S3_BUCKET/$folder/"
    aws s3 sync "$SRC" "$DEST"
done
echo -e "backup complete\n-----"