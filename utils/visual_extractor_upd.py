import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt


class VisualFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=300):
        super(VisualFeatureExtractor, self).__init__()

        self.resnet = models.resnet18(weights='DEFAULT')
        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()

        # linear projection
        self.projector = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = self.projector(x)
        return x


class FaceExtractor:
    def __init__(self, feature_extractor, device='cpu'):
        self.mtcnn = MTCNN()
        self.feature_extractor = feature_extractor
        self.device = device
        ##notgud duplicate
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def extract_faces_embedding(self, frame):

        ##notgud
        if isinstance(frame, torch.Tensor):
            frame = transforms.ToPILImage()(frame.squeeze(0))

        # find faces
        boxes, faces_confs = self.mtcnn.detect([frame])
        boxes = boxes[0]
        if faces_confs is not None and boxes is not None:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            total_area = sum(areas)
            # weight embeddings by areas of faces
            weights = [area / total_area for area in areas]

            face_embedding = torch.zeros(self.feature_extractor.projector.out_features).to(self.device)

            for i, box in enumerate(boxes):
                face = frame.crop((box[0], box[1], box[2], box[3]))
                face_tensor = self.img_transform(face).unsqueeze(0).to(self.device)
                face_emb = self.feature_extractor(face_tensor).squeeze(0)

                face_embedding += face_emb * weights[i]

            return face_embedding
        else:
            # faces wasn't found
            return torch.zeros(self.feature_extractor.projector.out_features).to(self.device)


class VideoFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=300, device='cpu'):
        super(VideoFeatureExtractor, self).__init__()
        self.visual_feature_extractor = VisualFeatureExtractor(feature_dim)
        self.face_extractor = FaceExtractor(self.visual_feature_extractor, device=device)
        self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def forward(self, frames):
        scene_embeddings = []
        face_embeddings = []

        # scene, faces embeds for each frame
        for frame in frames:

            frame_emb = self.visual_feature_extractor(self.img_transform(frame).unsqueeze(0))
            scene_embeddings.append(frame_emb)

            face_emb = self.face_extractor.extract_faces_embedding(self.img_transform(frame).unsqueeze(0))
            face_embeddings.append(face_emb)

        # max pooling
        scene_embeddings = torch.stack(scene_embeddings)
        scene_embeddings = scene_embeddings.permute(1, 0, 2).squeeze(0)
        scene_embeddings = torch.max(scene_embeddings, dim=0)[0]

        face_embeddings = torch.stack(face_embeddings)
        face_embeddings = torch.max(face_embeddings, dim=0)[0]

        final_embedding = torch.cat((scene_embeddings, face_embeddings), dim=0)
        return final_embedding


def get_pil_frames(video_path, num_frames=15, device='cpu'):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Video contains fewer than {num_frames} frames.")

    return frames


def display_frame(frame):
    plt.imshow(frame)
    plt.axis('off')
    plt.show()


def main():
    feature_dim = 300
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    video_feature_extractor = VideoFeatureExtractor(feature_dim=feature_dim, device=device).to(device)

    video_path = "dia3_utt0.mp4"
    frames = get_pil_frames(video_path)

    with torch.no_grad():
        embedding = video_feature_extractor(frames)
    print("Embedding shape:", embedding.shape)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    # model = torch.load('state_vggface2_enet2.pt')
    # model.eval()
    # main()
