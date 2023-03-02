# python run_qnt.py data/LibriTTS
# python run_g2p.py data/LibriTTS
python run_train.py yaml=config/LibriTTS/ar.yml
# python run_train.py yaml=config/LibriTTS/nar.yml
# python run_export.py zoo/ar.pt yaml=config/LibriTTS/ar.yml
# python run_export.py zoo/nar.pt yaml=config/LibriTTS/nar.yml
# python run.py "Well" "/home/liuhaozhe/voice_cloning_project/collected_audios/recorded_audios/liuhaozhe/liuhaozhe_text1.m4a" 