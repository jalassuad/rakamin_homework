## Apa itu Git dan Fungsinya dalam Kolaborasi
Git adalah sistem kontrol versi (version control system) yang digunakan untuk melacak perubahan dalam file, terutama kode program, secara efisien dan terstruktur. Dalam konteks kolaborasi, Git memungkinkan banyak orang bekerja pada proyek yang sama tanpa saling menimpa pekerjaan satu sama lain. Setiap kontributor dapat membuat perubahan di cabang (branch) masing-masing, lalu menggabungkannya (merge) ke versi utama setelah ditinjau. Ini sangat penting dalam tim data science, di mana skrip, model, dan dokumentasi sering diperbarui dan diuji secara paralel.

## Tiga Perintah Dasar Git dan Fungsinya

1. git init digunakan untuk menginisialisasi repository Git baru di dalam folder proyek, sehingga Git mulai melacak perubahan file di sana.
2. git add berfungsi untuk menambahkan file atau perubahan tertentu ke staging area, yaitu tempat sementara sebelum perubahan benar-benar disimpan.
3. git commit digunakan untuk menyimpan snapshot perubahan yang telah distage ke dalam repository, lengkap dengan pesan deskriptif agar mudah dilacak di masa depan.