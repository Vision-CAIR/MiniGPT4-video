# Excute this code in your terminal
# export REPLICATE_API_TOKEN=r8_SPtanEeudDQhUaRxIh4ftEzh7Sc91fr06j7P5
import replicate

input = {
    "video_path": "https://replicate.delivery/pbxt/Ki1Xy9IzGUX16CXvlMU1f9VYq89OpJk7hihhBR0CjScxp6so/Great%20white%20shark%20swims%20into%20cage.mp4",
    "question":"Give a brief description of the video",
    "add_subtitles": True,
}

output = replicate.run(
    "camenduru/minigpt4-video:5679342473d4fd99cf75e140a403e6463f8d5cdc324525783e0d7e35cf27f68b",
    input=input
)
print(output)
#=> "This video is about a man and sharks in the ocean. The m...