import os
import sys
import threading
from PIL import Image

import customtkinter as ctk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Cấu hình giao diện CustomTkinter hiện đại
ctk.set_appearance_mode("Dark")           # Chế độ nền tối sang trọng
ctk.set_default_color_theme("blue")      # Chủ đề màu xanh công nghệ cao

# 1. Định nghĩa kiến trúc mạng CNN chuẩn Sigmoid từ Notebook
class ChestXRayBinaryNet(nn.Module):
    def __init__(self):
        super(ChestXRayBinaryNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class PneumoniaModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Chẩn Đoán Viêm Phổi Từ Ảnh X-Quang")
        self.geometry("1060x730")
        self.minsize(1060, 730)
        self.resizable(False, False) # Giữ cố định kích thước khung giao diện không bị co giãn khi nạp ảnh

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets")
        
        # Đổi logo icon lông vũ mặc định của Tkinter thành icon phổi trên thanh tiêu đề Windows
        ico_file = os.path.join(self.assets_dir, "lungs_icon.ico")
        if os.path.exists(ico_file):
            try:
                self.iconbitmap(ico_file)
            except Exception:
                pass
        
        # Tự động tìm thư mục dữ liệu (trong Chest_Xray_Pneumonia hoặc ở thư mục cha CNN_Train)
        cand_list = [
            os.path.join(self.base_dir, 'chest_xray', 'chest_xray'),
            os.path.join(self.base_dir, 'chest_xray'),
            os.path.join(os.path.dirname(self.base_dir), 'chest_xray', 'chest_xray'),
            os.path.join(os.path.dirname(self.base_dir), 'chest_xray')
        ]
        self.data_dir = self.base_dir
        for c in cand_list:
            if os.path.exists(os.path.join(c, 'train')):
                self.data_dir = c
                break

        self.train_dir = os.path.join(self.data_dir, 'train')
        
        # Đường dẫn lưu / nạp model
        saved_models_dir = os.path.join(self.base_dir, "saved_models")
        os.makedirs(saved_models_dir, exist_ok=True)
        self.model_save_path = os.path.join(saved_models_dir, "best_chest_xray_model.pth")
        
        self.transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = ChestXRayBinaryNet().to(self.device)
        self.is_model_loaded = False
        self.load_model_weights()
        self.current_image_path = None

        self.build_ui()

    def load_model_weights(self):
        if os.path.exists(self.model_save_path):
            try:
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                self.model.eval()
                self.is_model_loaded = True
            except Exception as e:
                print(f"Lỗi tải model: {e}")
                self.is_model_loaded = False
        else:
            self.is_model_loaded = False

    def build_ui(self):
        # 1. Header cao cấp với Logo phổi y tế
        header_card = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=0, height=65)
        header_card.pack(fill="x", padx=0, pady=0)
        
        png_icon_path = os.path.join(self.assets_dir, "lungs_icon.png")
        if os.path.exists(png_icon_path):
            try:
                pil_icon = Image.open(png_icon_path)
                self.header_logo = ctk.CTkImage(light_image=pil_icon, dark_image=pil_icon, size=(36, 36))
                lbl_logo = ctk.CTkLabel(header_card, image=self.header_logo, text="")
                lbl_logo.pack(side="left", padx=(25, 10), pady=12)
            except Exception:
                pass
        
        lbl_brand = ctk.CTkLabel(header_card, text="HỆ THỐNG CHẨN ĐOÁN VIÊM PHỔI TỪ ẢNH X-QUANG", 
                                font=ctk.CTkFont(family="Arial", size=16, weight="bold"), text_color="#38BDF8")
        lbl_brand.pack(side="left", padx=(0, 20), pady=16)

        lbl_device = ctk.CTkLabel(header_card, text=f"Thiết bị: {str(self.device).upper()}", 
                                 font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                                 fg_color="#1E293B", text_color="#94A3B8", corner_radius=8, padx=12, pady=4)
        lbl_device.pack(side="right", padx=25, pady=16)

        # 2. Main Container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=15)
        main_frame.grid_columnconfigure(0, weight=5)
        main_frame.grid_columnconfigure(1, weight=6)
        main_frame.grid_rowconfigure(0, weight=1)

        # ==========================================
        # CỘT TRÁI: HÌNH ẢNH X-QUANG
        # ==========================================
        left_card = ctk.CTkFrame(main_frame, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_card.grid_rowconfigure(1, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        lbl_left_title = ctk.CTkLabel(left_card, text="HÌNH ẢNH X-QUANG PHỔI", 
                                     font=ctk.CTkFont(family="Arial", size=13, weight="bold"), text_color="#E2E8F0")
        lbl_left_title.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 5))

        # Khung preview ảnh
        self.image_preview_box = ctk.CTkFrame(left_card, fg_color="#0F172A", corner_radius=12, border_width=1, border_color="#1E293B")
        self.image_preview_box.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.image_preview_box.grid_rowconfigure(0, weight=1)
        self.image_preview_box.grid_columnconfigure(0, weight=1)

        self.lbl_image = ctk.CTkLabel(self.image_preview_box, text="Chưa chọn ảnh X-quang nào.\n\nNhấn nút bên dưới để chọn ảnh và tự động chẩn đoán ngay.", 
                                     font=ctk.CTkFont(size=13), text_color="#64748B", justify="center")
        self.lbl_image.grid(row=0, column=0, padx=20, pady=20)

        # Nút chọn ảnh (chọn xong sẽ tự động chẩn đoán ngay)
        btn_browse = ctk.CTkButton(left_card, text="📁 Chọn Ảnh X-Quang Từ Máy Tính", 
                                  font=ctk.CTkFont(size=13, weight="bold"), height=42, corner_radius=10,
                                  fg_color="#0284C7", hover_color="#0369A1", cursor="hand2", command=self.choose_and_predict_image)
        btn_browse.grid(row=2, column=0, sticky="ew", padx=20, pady=(5, 18))

        # ==========================================
        # CỘT PHẢI: KẾT QUẢ CHẨN ĐOÁN & NHẬT KÝ LOG CHUNG
        # ==========================================
        right_container = ctk.CTkFrame(main_frame, fg_color="transparent")
        right_container.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        right_container.grid_rowconfigure(0, weight=0)
        right_container.grid_rowconfigure(1, weight=1)
        right_container.grid_columnconfigure(0, weight=1)

        # PHẦN 1: THẺ KẾT QUẢ CHẨN ĐOÁN
        result_card = ctk.CTkFrame(right_container, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        result_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        lbl_res_header = ctk.CTkLabel(result_card, text="KẾT QUẢ CHẨN ĐOÁN AI", 
                                     font=ctk.CTkFont(family="Arial", size=13, weight="bold"), text_color="#E2E8F0")
        lbl_res_header.pack(anchor="w", padx=20, pady=(14, 8))

        self.badge_status = ctk.CTkLabel(result_card, text="CHƯA CHỌN ẢNH", 
                                         font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                                         fg_color="#0F172A", text_color="#94A3B8", corner_radius=10, height=42)
        self.badge_status.pack(fill="x", padx=20, pady=(0, 12))

        # Thanh Progress Bar xác suất Viêm Phổi
        row_p = ctk.CTkFrame(result_card, fg_color="transparent")
        row_p.pack(fill="x", padx=20, pady=(0, 4))
        ctk.CTkLabel(row_p, text="Xác suất Viêm Phổi (Pneumonia):", font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(side="left")
        self.lbl_prob_p = ctk.CTkLabel(row_p, text="0.0%", font=ctk.CTkFont(family="Consolas", size=13, weight="bold"), text_color="#F87171")
        self.lbl_prob_p.pack(side="right")

        self.progress_p = ctk.CTkProgressBar(result_card, height=10, corner_radius=5, fg_color="#0F172A", progress_color="#EF4444")
        self.progress_p.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_p.set(0)

        # Thanh Progress Bar xác suất Phổi Bình Thường
        row_n = ctk.CTkFrame(result_card, fg_color="transparent")
        row_n.pack(fill="x", padx=20, pady=(0, 4))
        ctk.CTkLabel(row_n, text="Xác suất Bình Thường (Normal):", font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(side="left")
        self.lbl_prob_n = ctk.CTkLabel(row_n, text="0.0%", font=ctk.CTkFont(family="Consolas", size=13, weight="bold"), text_color="#4ADE80")
        self.lbl_prob_n.pack(side="right")

        self.progress_n = ctk.CTkProgressBar(result_card, height=10, corner_radius=5, fg_color="#0F172A", progress_color="#22C55E")
        self.progress_n.pack(fill="x", padx=20, pady=(0, 16))
        self.progress_n.set(0)

        # PHẦN 2: LOG CHUNG DÀI HẾT BÊN DƯỚI (Nút Train thu nhỏ trên đầu Log)
        log_card = ctk.CTkFrame(right_container, fg_color="#1E293B", corner_radius=16, border_width=1, border_color="#334155")
        log_card.grid(row=1, column=0, sticky="nsew")
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)

        # Header của Log: Gồm Tiêu đề bên trái và Nút Train thu nhỏ bên phải
        log_header_frame = ctk.CTkFrame(log_card, fg_color="transparent")
        log_header_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))

        lbl_log_title = ctk.CTkLabel(log_header_frame, text="NHẬT KÝ HỆ THỐNG (LOG)", 
                                     font=ctk.CTkFont(family="Arial", size=13, weight="bold"), text_color="#E2E8F0")
        lbl_log_title.pack(side="left")

        # Nút Huấn luyện lại thu nhỏ nằm trên đầu log
        self.btn_retrain = ctk.CTkButton(log_header_frame, text="🔄 Huấn Luyện Lại (Re-Train)", 
                                         font=ctk.CTkFont(size=12, weight="bold"), height=30, corner_radius=8,
                                         fg_color="#EA580C", hover_color="#C2410C", cursor="hand2", 
                                         command=self.start_retrain_thread)
        self.btn_retrain.pack(side="right")

        # Log Textbox mở rộng hết phần còn lại của cột phải
        self.txt_log = ctk.CTkTextbox(log_card, fg_color="#0F172A", text_color="#38BDF8", 
                                      font=ctk.CTkFont(family="Consolas", size=12), corner_radius=10)
        self.txt_log.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))

        # Khởi tạo log ban đầu không dấu
        self.log("He thong da san sang.")
        self.log(f"Thiet bi: {str(self.device).upper()}")
        self.log(f"Thu muc data train: {self.train_dir}")
        if self.is_model_loaded:
            self.log("Trang thai model: Da nap weights 'best_chest_xray_model.pth' thanh cong.")
        else:
            self.log("CHU Y: Chua co file model weights da train ('best_chest_xray_model.pth').")
            self.log("=> Mo hinh hien tai dang co trong so ngau nhien (xac suat se loan quanh 50%).")
            self.log("=> Hay bam nut 'Huan Luyen Lai (Re-Train)' o tren de model hoc du lieu!")
        self.log("-" * 55)

    def log(self, text):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def choose_and_predict_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh phim X-quang",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        self.current_image_path = path
        filename = os.path.basename(path)

        try:
            # 1. Hiển thị ảnh xem trước
            pil_img = Image.open(path)
            self.ctk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(380, 380))
            self.lbl_image.configure(image=self.ctk_image, text="")

            # 2. Tự động chẩn đoán ngay lập tức
            raw_img = pil_img.convert("RGB")
            img_tensor = self.transform_test(raw_img).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output_logit = self.model(img_tensor)
                prob_pneumonia = torch.sigmoid(output_logit).item() * 100
                prob_normal = 100.0 - prob_pneumonia

            ratio_p = prob_pneumonia / 100.0
            ratio_n = prob_normal / 100.0

            # Cập nhật thanh đo và %
            self.progress_p.set(ratio_p)
            self.progress_n.set(ratio_n)
            self.lbl_prob_p.configure(text=f"{prob_pneumonia:.2f}%")
            self.lbl_prob_n.configure(text=f"{prob_normal:.2f}%")

            # Kết luận và ghi log không dấu
            if prob_pneumonia >= 50.0:
                self.badge_status.configure(text="KẾT QUẢ: VIÊM PHỔI (PNEUMONIA)", fg_color="#7F1D1D", text_color="#FCA5A5")
                result_label = f"viem phoi ({prob_pneumonia:.1f}%)"
            else:
                self.badge_status.configure(text="KẾT QUẢ: PHỔI BÌNH THƯỜNG (NORMAL)", fg_color="#14532D", text_color="#86EFAC")
                result_label = f"binh thuong ({prob_normal:.1f}%)"

            # Ghi log chuẩn định dạng không dấu: ten_file - du doan - ket qua
            self.log(f"{filename} - du doan - {result_label}")

        except Exception as e:
            self.log(f"Loi xu ly anh {filename}: {e}")
            messagebox.showerror("Lỗi", f"Có lỗi khi xử lý ảnh:\n{e}")

    def start_retrain_thread(self):
        if not os.path.exists(self.train_dir):
            messagebox.showerror("Lỗi", f"Không tìm thấy thư mục dữ liệu train tại:\n{self.train_dir}")
            return

        self.btn_retrain.configure(state="disabled", text="Đang train...")
        t = threading.Thread(target=self.run_retrain)
        t.daemon = True
        t.start()

    def run_retrain(self):
        try:
            epochs = 3
            batch_size = 32

            self.log("-" * 55)
            self.log("BAT DAU HUAN LUYEN LAI (RE-TRAIN)")
            self.log(f"Thu muc: {self.train_dir}")
            
            train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.transform_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            class_counts = [train_dataset.targets.count(i) for i in range(len(train_dataset.classes))]
            normal_cnt = class_counts[0]
            pneumonia_cnt = max(class_counts[1], 1)
            pos_weight = torch.tensor([normal_cnt / pneumonia_cnt]).to(self.device)

            self.log(f"Tong so anh: {len(train_dataset)} (Normal: {normal_cnt}, Pneumonia: {pneumonia_cnt})")
            self.log(f"Trong so pos_weight: {pos_weight.item():.3f}")

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

            for ep in range(1, epochs + 1):
                self.model.train()
                total_loss, correct, total = 0, 0, 0
                for imgs, lbls in train_loader:
                    imgs = imgs.to(self.device)
                    lbls = lbls.float().unsqueeze(1).to(self.device)

                    optimizer.zero_grad()
                    outs = self.model(imgs)
                    loss = criterion(outs, lbls)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * imgs.size(0)
                    preds = (torch.sigmoid(outs) >= 0.5).float()
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)

                train_acc = correct / total
                self.log(f"Epoch {ep:02d}/{epochs:02d} - Loss: {total_loss/total:.4f} | Acc: {train_acc*100:.2f}%")

            torch.save(self.model.state_dict(), self.model_save_path)
            self.model.eval()
            self.log("Luu va nap mo hinh moi thanh cong!")
            self.log("-" * 55)
            messagebox.showinfo("Hoàn tất", "Đã huấn luyện lại xong trên dữ liệu mới!\nBạn có thể chọn ảnh để thử ngay.")

        except Exception as e:
            self.log(f"Loi khi train: {e}")
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {e}")
        finally:
            self.btn_retrain.configure(state="normal", text="🔄 Huấn Luyện Lại (Re-Train)")

if __name__ == '__main__':
    app = PneumoniaModernApp()
    app.mainloop()
