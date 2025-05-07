---
layout: article
title: Document - Writing Posts
tags:
  - GitHub
mathjax: true
---
> [!info] 文件创建时间
> 2025-05-07,15:53



> [!info] 提交任务
> 1. 进入相关项目的文件夹目录底下（拖拽文件夹到终端）
> 2. 检查是否连接成功 `ssh -T git@github.com`
> 3. 显示远程仓库地址：`git remote -v`
> 4. 上传文件：`git add . / git add xxx.md`
> 5. 说明更换版本`git commit -m “版本说明文字”`
> 6. 提交文件 `git push`


## Github 检测连接
```bash
ssh -T git@github.com
git remote -v
```

## 修改文件提交任务
```bash
git add . / git add xxx.md
git commit -m “版本说明文字”
git push
```

- 直接 `git add .` 全部提交上去，其实github 会自动检查哪些文件修改了，最终只会提交修改了的文件。