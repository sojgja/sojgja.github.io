1. tạo repository trên github sojgja.github.io và tạo branch main để deploy
2. tạo folders www
3. clone mkdocs-material về thư mục www
4. copy toàn bộ file và folder trong mkdocs-material ra thư mục gốc
5. vào đường dẫn và thiết lập
	https://github.com/sojgja/sojgja.github.io/settings/actions
	chọn Read and write permissions
6. vào đường dẫn và thiết lập
	https://github.com/sojgja/sojgja.github.io/settings/pages
	chọn deploy from a branch
	chọn gh-pages
7. chỉnh sửa các file cần thiết 
	https://github.com/sojgja/sojgja.github.io/blob/dev/.github/workflows/ci.yml
      - run: pip install mkdocs-material mkdocs-minify-plugin
8. push code lên nhánh main để thực hiện deploy