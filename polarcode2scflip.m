% =========================================================================
% STANDARD SC FLIP DECODER (EPFL PAPER IMPLEMENTATION)
% Direct copy & run (Full Single File)
% =========================================================================
clear; clc; close all;

%% 1. CẤU HÌNH THAM SỐ
N = 256;                        
K_info = 128;                   
crc_poly = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1]; % CRC-16-CCITT
r = length(crc_poly) - 1;       
K_total = K_info + r;           

EbNo_dB = 1.0:0.5:3.5;          
max_attempts = 32;              % Số lần lật thử tối đa (T_max) theo bài báo
max_frame_errors = 30;          
max_frames = 2000;              

R_eff = K_info / N;             
design_snr_db = 2.0;            
frozen_mask = construct_polar_code(N, K_total, design_snr_db);

%% 2. MÔ PHỎNG MONTE CARLO
FER_SC   = zeros(1, length(EbNo_dB));
FER_SCF  = zeros(1, length(EbNo_dB));

for i = 1:length(EbNo_dB)
    ebno = EbNo_dB(i);
    snr_lin = 10^(ebno/10);
    sigma = sqrt(1 / (2 * R_eff * snr_lin)); 
    
    err_sc = 0;
    err_scf = 0;
    frames = 0;
    
    fprintf('--> Đang mô phỏng Eb/N0 = %.2f dB...\n', ebno);
    
    while (err_scf < max_frame_errors) && (frames < max_frames)
        frames = frames + 1;
        
        % Tạo dữ liệu + CRC
        u_info = randi([0 1], 1, K_info);
        u_crc = generate_crc(u_info, crc_poly);
        u_payload = [u_info, u_crc];
        
        u = zeros(1, N);
        u(frozen_mask == 1) = u_payload;
        
        % Mã hóa Polar chuẩn
        x = polar_encode(u);
        
        % Kênh truyền AWGN
        s = 1 - 2*x; 
        y = s + sigma * randn(1, N);
        llr_channel = 2 * y / (sigma^2);
        
        % 1. Giải mã SC tiêu chuẩn
        [u_hat_sc, ~] = polar_sc_decode(llr_channel, frozen_mask, -1);
        if ~check_crc(u_hat_sc(frozen_mask == 1), crc_poly, K_info)
            err_sc = err_sc + 1;
        end
        
        % 2. Giải mã SC Flip chuẩn theo bài báo EPFL
        u_hat_scf = sc_flip_decode(llr_channel, frozen_mask, crc_poly, K_info, max_attempts);
        if ~check_crc(u_hat_scf(frozen_mask == 1), crc_poly, K_info)
            err_scf = err_scf + 1;
        end
    end
    
    FER_SC(i)  = err_sc / frames;
    FER_SCF(i) = err_scf / frames;
    fprintf('    Frames: %d | FER SC: %.4e | FER SCF: %.4e\n', frames, FER_SC(i), FER_SCF(i));
end

%% 3. VẼ BIỂU ĐỒ
figure('Color', [1 1 1]);
semilogy(EbNo_dB, FER_SC, '-ro', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'SC Decoder');
hold on;
semilogy(EbNo_dB, FER_SCF, '-bs', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'SC Flip (EPFL)');
grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Frame Error Rate (FER)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Polar Code Simulation (N=%d, K=%d)', N, K_info));
legend('Location', 'southwest', 'FontSize', 10);
ylim([1e-4 1]);

% =========================================================================
% HÀM CỤC BỘ (LOCAL FUNCTIONS)
% =========================================================================

function best_u_hat = sc_flip_decode(llr, frozen_mask, crc_poly, K_info, max_attempts)
    % Lần chạy đầu tiên: SC tiêu chuẩn (không lật bit nào -> truyền vào -1)
    [u_hat, decision_llrs] = polar_sc_decode(llr, frozen_mask, -1);
    
    % Nếu vượt qua kiểm tra CRC ngay lần đầu thì trả về kết quả luôn
    if check_crc(u_hat(frozen_mask == 1), crc_poly, K_info)
        best_u_hat = u_hat;
        return;
    end
    
    % Sắp xếp các bit thông tin (chỉ số gốc) theo độ tin cậy tăng dần |LLR|
    info_indices = find(frozen_mask == 1);
    [~, sort_order] = sort(abs(decision_llrs(info_indices)), 'ascend');
    sorted_info_indices = info_indices(sort_order);
    
    % Tiến hành lật thử từng vị trí có độ tin cậy thấp nhất (T_max lần)
    for T = 1:min(max_attempts, length(sorted_info_indices))
        flip_pos = sorted_info_indices(T);
        
        % Chạy lại bộ giải mã SC với 1 vị trí bit được định sẵn là sẽ lật ngược kết quả cứng
        [u_trial, ~] = polar_sc_decode(llr, frozen_mask, flip_pos);
        
        if check_crc(u_trial(frozen_mask == 1), crc_poly, K_info)
            best_u_hat = u_trial; 
            return; % Thành công, thoát hàm sớm
        end
    end
    
    % Nếu thử hết T_max lần vẫn lỗi, trả về kết quả của lần giải mã SC đầu tiên
    best_u_hat = u_hat;
end

function [u_hat, decision_llrs] = polar_sc_decode(llr, frozen_mask, flip_pos)
    N = length(llr);
    n = log2(N);
    
    % Khởi tạo cấu trúc bảng LLR và Bit (Đã đồng bộ hóa chỉ số)
    L = zeros(n+1, N);
    B = zeros(n+1, N);
    decision_llrs = zeros(1, N);
    
    % Đưa LLR kênh truyền vào tầng đầu tiên theo Bit-Reversal
    rev_idx = bitrevorder(0:N-1) + 1;
    L(1, :) = llr(rev_idx);
    
    % Tạo mặt nạ frozen đã được đảo bit tương ứng với tầng lá cây giải mã
    frozen_mask_rev = frozen_mask(rev_idx);
    u_hat_rev = zeros(1, N);
    
    for i = 1:N
        % Bước 1: Cập nhật LLR từ trên xuống dưới (Forward Path)
        for stage = 1:n
            node_size = 2^(n - stage + 1);
            half_size = node_size / 2;
            idx_in_node = mod(i - 1, node_size);
            
            if idx_in_node == 0
                parent = floor((i - 1) / node_size) * node_size + 1;
                x = L(stage, parent : parent + half_size - 1);
                y = L(stage, parent + half_size : parent + node_size - 1);
                % Hàm f: công thức (3) trong bài báo
                L(stage+1, parent : parent + half_size - 1) = sign(x).*sign(y).*min(abs(x), abs(y));
            elseif idx_in_node == half_size
                parent = floor((i - 1) / node_size) * node_size + 1;
                x = L(stage, parent : parent + half_size - 1);
                y = L(stage, parent + half_size : parent + node_size - 1);
                u1 = B(stage+1, parent : parent + half_size - 1);
                % Hàm g: công thức (4) trong bài báo g = (-1)^u1 * x + y
                L(stage+1, parent + half_size : parent + node_size - 1) = ((-1).^u1).*x + y;
            end
        end
        
        % Bước 2: Quyết định cứng tại tầng lá (Tầng n+1)
        decision_llrs(i) = L(n+1, i);
        if frozen_mask_rev(i) == 0
            u_hat_rev(i) = 0; % Bit đóng băng cố định bằng 0
        else
            hard_dec = (L(n+1, i) < 0);
            
            % Ánh xạ vị trí i trong mảng đảo bit về chỉ số gốc để check flip_pos
            orig_idx = rev_idx(i); 
            if orig_idx == flip_pos
                u_hat_rev(i) = ~hard_dec; % Lật bit nếu trùng với vị trí lỗi giả định
            else
                u_hat_rev(i) = hard_dec;
            end
        end
        B(n+1, i) = u_hat_rev(i);
        
        % Bước 3: Truyền ngược bit quyết định lên (Backward Path) - Phương trình (5) & (6)
        for stage = n:-1:1
            node_size = 2^(n - stage + 1);
            half_size = node_size / 2;
            idx_in_node = mod(i - 1, node_size);
            
            if idx_in_node >= half_size
                parent = floor((i - 1) / node_size) * node_size + 1;
                u1 = B(stage+1, parent : parent + half_size - 1);
                u2 = B(stage+1, parent + half_size : parent + node_size - 1);
                B(stage, parent : parent + node_size - 1) = [mod(u1 + u2, 2), u2];
            else
                break;
            end
        end
    end
    
    % Trả chuỗi quyết định về thứ tự tuyến tính ban đầu để xử lý CRC ngoài hàm main
    u_hat = zeros(1, N);
    u_hat(rev_idx) = u_hat_rev;
    
    % Khôi phục mảng decision_llrs về thứ tự tuyến tính tương ứng các bit u gốc
    decision_llrs_ordered = zeros(1, N);
    decision_llrs_ordered(rev_idx) = decision_llrs;
    decision_llrs = decision_llrs_ordered;
end

function x = polar_encode(u)
    N = length(u);
    n = log2(N);
    F = [1 0; 1 1];
    G = 1;
    for i = 1:n
        G = kron(G, F);
    end
    x = mod(u * G, 2);
end

function frozen_mask = construct_polar_code(N, K, design_snr_db)
    n = log2(N);
    z = zeros(1, N);
    z(1) = exp(-10^(design_snr_db/10));
    
    for i = 1:n
        u = 2^(i-1);
        for j = 1:u
            T = z(j);
            z(j)   = min(2*T - T^2, 1 - 1e-10);
            z(j+u) = max(T^2, 1e-10);
        end
    end
    
    [~, idx] = sort(z, 'ascend');
    frozen_mask = zeros(1, N);
    frozen_mask(idx(1:K)) = 1;
end

function u_crc = generate_crc(info, poly)
    crc_len = length(poly) - 1; 
    msg = [info, zeros(1, crc_len)];
    for i = 1:length(info)
        if msg(i) == 1
            msg(i:i+crc_len) = bitxor(msg(i:i+crc_len), poly);
        end
    end
    u_crc = msg(end-crc_len+1:end);
end

function is_valid = check_crc(payload, poly, K_info)
    crc_len = length(poly) - 1; 
    msg = payload;
    for i = 1:K_info
        if msg(i) == 1
            msg(i:i+crc_len) = bitxor(msg(i:i+crc_len), poly);
        end
    end
    is_valid = all(msg(K_info+1:end) == 0);
end
