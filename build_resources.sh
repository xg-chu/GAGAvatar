echo "In order to run this tool, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

mkdir -p assets
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/FLAME_with_eye.pt -O ./assets/FLAME_with_eye.pt
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/canonical.obj -O ./assets/canonical.obj
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/assets/GAGAvatar.pt -O ./assets/GAGAvatar.pt
wget https://huggingface.co/xg-chu/GAGAvatar/resolve/main/demos.tar -O ./demos.tar
tar -xvf demos.tar -C demos/
rm -r demos.tar


