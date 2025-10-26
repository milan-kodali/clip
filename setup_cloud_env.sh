# set up cloud box
# run: `source setup_cloud_env.sh`

# check if uv is installed; install it if not
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# check if .venv dir exists; create a venv if not
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Set Git identity
echo "setting up git config"
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"
echo -e "git config complete\n-----"

# set default shell working dir to clip repo
echo "setting default shell working dir"
echo 'cd /workspace/clip' >> ~/.bashrc
source ~/.bashrc
echo -e "default shell working dir set\n-----"

# install aws cli
echo "installing aws cli"
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws
aws --version
echo -e "aws cli setup complete\n-----"

# download data from s3
echo "downloading data, checkpoints, and logs from s3"
FOLDERS=("clip_data")
for folder in ${FOLDERS[@]}; do
    SRC="s3://$S3_BUCKET/$folder/"
    DEST="./.cache/$folder/"
    mkdir -p "$DEST"
    aws s3 sync "$SRC" "$DEST"
done
echo -e "download complete\n-----"