import React, { useState } from 'react';

const VideoRetrievalDashboard = () => {
  const [selected_query_type, setSelectedQueryType] = useState('textual_kis');
  const [query_text_input, setQueryTextInput] = useState('');
  const [name_output_input, setNameOutputInput] = useState('test');
  const [question_text_input, setQuestionTextInput] = useState('');
  const [is_loading, setIsLoading] = useState(false);
  const [top_100_results, setTop100Results] = useState([]);
  
  // Lưu thông tin của video đang được click để mở Popup
  const [playing_video, setPlayingVideo] = useState(null);

  const handleSearch = async () => {
    setIsLoading(true);
    try {
      const payload = {
        type: selected_query_type,
        query: query_text_input,
        question: selected_query_type === 'q_a' ? question_text_input : null,
        name_output: name_output_input
      };

      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await response.json();
      setTop100Results(data.list_search_results);
    } catch (error) {
      console.error("Lỗi khi tìm kiếm:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async () => {
    console.log("Dữ liệu chuẩn bị xuất:", top_100_results);
    alert(`Đã chuẩn bị xuất ${top_100_results.length} kết quả!`);
  };

  // Hàm tính toán khoảng thời gian (trước 5s và sau 5s) dựa vào frame_id
  const getVideoTimeRange = (frame_id) => {
    const FPS = 25; // Giả định tốc độ video là 25 khung hình/giây
    const target_second = parseInt(frame_id) / FPS;
    
    // Tính thời gian bắt đầu (dùng Math.max để không bị số âm nếu sự kiện ở đầu video)
    const start_time = Math.max(0, target_second - 5).toFixed(1);
    
    // Tính thời gian kết thúc
    const end_time = (target_second + 5).toFixed(1);
    
    return `${start_time},${end_time}`;
  };

  return (
    <div style={{ 
      display: 'flex', flexDirection: 'column', height: '100vh', 
      backgroundColor: '#f9fafb', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      
      {/* HEADER */}
      <header style={{ padding: '16px 32px', backgroundColor: '#ffffff', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #e5e7eb' }}>
        <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '600', color: '#111827' }}>
          AIC 2026 <span style={{ color: '#6b7280', fontWeight: '400' }}>| Video Retrieval</span>
        </h1>
        <select 
          value={selected_query_type} 
          onChange={(e) => setSelectedQueryType(e.target.value)}
          style={{ padding: '8px 16px', fontSize: '14px', borderRadius: '8px', border: '1px solid #d1d5db', outline: 'none', cursor: 'pointer' }}
        >
          <option value="textual_kis">1. Textual KIS (Mô tả)</option>
          <option value="q_a">2. Q&A (Hỏi Đáp)</option>
          <option value="trake">3. TRAKE (Chuỗi sự kiện)</option>
        </select>
      </header>

      {/* KHU VỰC TÌM KIẾM */}
      <div style={{ padding: '32px', backgroundColor: '#ffffff', borderBottom: '1px solid #e5e7eb', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <textarea 
          placeholder="Nhập mô tả sự kiện..." value={query_text_input} onChange={(e) => setQueryTextInput(e.target.value)}
          rows={2} 
          style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '15px', resize: 'vertical', outline: 'none', boxSizing: 'border-box' }}
        />
        
        {selected_query_type === 'q_a' && (
          <input 
            type="text" placeholder="Nhập câu hỏi..." value={question_text_input} onChange={(e) => setQuestionTextInput(e.target.value)}
            style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '15px', outline: 'none', boxSizing: 'border-box' }}
          />
        )}
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input 
            type="text" 
            placeholder="Tên file xuất (e.g. test)..." 
            value={name_output_input} 
            onChange={(e) => setNameOutputInput(e.target.value)}
            style={{ padding: '10px 16px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '14px', outline: 'none', width: '220px' }}
          />
          <button 
            onClick={handleSearch} disabled={is_loading}
            style={{ padding: '10px 24px', backgroundColor: is_loading ? '#9ca3af' : '#111827', color: '#ffffff', border: 'none', borderRadius: '8px', fontSize: '14px', fontWeight: '500', cursor: is_loading ? 'not-allowed' : 'pointer' }}
          >
            {is_loading ? 'Đang phân tích...' : 'Truy xuất Top 100'}
          </button>
          <button 
            onClick={handleExport} disabled={top_100_results.length === 0}
            style={{ padding: '10px 24px', backgroundColor: top_100_results.length === 0 ? '#f3f4f6' : '#10b981', color: top_100_results.length === 0 ? '#9ca3af' : '#ffffff', border: 'none', borderRadius: '8px', fontSize: '14px', fontWeight: '500', cursor: top_100_results.length === 0 ? 'not-allowed' : 'pointer' }}
          >
            Xuất File CSV
          </button>
        </div>
      </div>

      {/* KHU VỰC HIỂN THỊ DANH SÁCH */}
      <div style={{ padding: '32px', overflowY: 'auto', flex: 1 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '24px' }}>
          {top_100_results.map((res, index) => (
            <div key={index} style={{ backgroundColor: '#ffffff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              
              {/* VÙNG HÌNH ẢNH */}
              <div 
                style={{ position: 'relative', cursor: 'pointer' }}
                onClick={() => setPlayingVideo(res)}
                title="Nhấn để xem video sự kiện"
              >
                <img 
                    src={`http://localhost:8000/api/images/${res.video_id}/${res.frame_id}`} 
                    alt={`Top ${index + 1}`} 
                    style={{width: '100%', height: '160px', objectFit: 'cover', backgroundColor: '#f3f4f6', display: 'block', transition: 'opacity 0.2s'}}
                    onMouseOver={e => e.currentTarget.style.opacity = 0.8}
                    onMouseOut={e => e.currentTarget.style.opacity = 1}
                />
                
                {/* Icon Play */}
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', backgroundColor: 'rgba(0,0,0,0.5)', color: 'white', borderRadius: '50%', width: '40px', height: '40px', display: 'flex', justifyContent: 'center', alignItems: 'center', pointerEvents: 'none' }}>
                  ▶
                </div>

                <div style={{ position: 'absolute', top: '8px', left: '8px', backgroundColor: 'rgba(17, 24, 39, 0.8)', color: 'white', padding: '4px 10px', borderRadius: '20px', fontSize: '12px', fontWeight: '600' }}>
                  #{index + 1}
                </div>
              </div>

              {/* Thông tin chi tiết */}
              <div style={{ padding: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>{res.video_id}</span>
                  <span style={{ fontSize: '12px', fontWeight: '600', color: '#059669', backgroundColor: '#d1fae5', padding: '2px 8px', borderRadius: '12px' }}>
                    {(res.matching_score * 100).toFixed(1)}%
                  </span>
                </div>
                <p style={{ margin: '0 0 12px 0', fontSize: '13px', color: '#6b7280' }}>Khung hình: <span style={{ color: '#111827', fontWeight: '500' }}>{res.frame_id}</span></p>
                {res.predicted_answer && (
                  <div style={{ marginTop: 'auto', padding: '8px 12px', backgroundColor: '#f3f4f6', borderRadius: '6px', fontSize: '13px', color: '#4b5563' }}>
                    <span style={{ fontWeight: '600', color: '#111827' }}>Đáp án:</span> {res.predicted_answer}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* POPUP MODAL PHÁT VIDEO */}
      {playing_video && (
        <div 
          style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.85)', zIndex: 9999, display: 'flex', justifyContent: 'center', alignItems: 'center', backdropFilter: 'blur(4px)' }}
          onClick={() => setPlayingVideo(null)} 
        >
          <div 
            style={{ width: '80%', maxWidth: '1000px', backgroundColor: '#000', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)' }}
            onClick={(e) => e.stopPropagation()} 
          >
            {/* Thanh tiêu đề Modal có hiển thị khoảng thời gian */}
            <div style={{ padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#111827', color: 'white' }}>
              <div>
                <span style={{ fontSize: '16px', fontWeight: '600' }}>{playing_video.video_id}</span>
                <span style={{ color: '#9ca3af', marginLeft: '12px', fontSize: '14px' }}>
                  Frame: {playing_video.frame_id} (Phát từ giây {Math.max(0, (parseInt(playing_video.frame_id) / 25) - 5).toFixed(1)} đến {(parseInt(playing_video.frame_id) / 25 + 5).toFixed(1)})
                </span>
              </div>
              <button 
                onClick={() => setPlayingVideo(null)}
                style={{ background: 'none', border: 'none', color: '#9ca3af', fontSize: '24px', cursor: 'pointer', padding: 0 }}
                title="Đóng (Phím Esc)"
              >
                ×
              </button>
            </div>
            
            {/* Trình phát Video có gọi hàm tính khoảng thời gian */}
            <video 
              src={`http://localhost:8000/api/videos/${playing_video.video_id}#t=${getVideoTimeRange(playing_video.frame_id)}`}
              controls
              autoPlay
              style={{ width: '100%', display: 'block', maxHeight: '70vh', outline: 'none' }}
            />
          </div>
        </div>
      )}

    </div>
  );
};

export default VideoRetrievalDashboard;