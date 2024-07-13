package utils

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/itchio/lzma"
)

func UnzipDataset(fileURL string, targetFolderURL string) error {
	file, err := zip.OpenReader(fileURL)
	if err != nil {
		return err
	}
	defer file.Close()
	file.RegisterDecompressor(0xe, lzma.NewReader)

	err = os.Mkdir(targetFolderURL, 0750)
	if err != nil && !os.IsExist(err) {
		return err
	}

	for _, f := range file.File {
		fmt.Printf("Unzipping %s:\n", f.Name)
		if f.FileInfo().IsDir() {
			fmt.Println("dir")
		}
		rs, err := f.Open()
		if err != nil {
			return err
		}

		dirPath := filepath.Dir(f.Name)
		err = os.MkdirAll(dirPath, 0750)
		if err != nil && !os.IsExist(err) {
			return err
		}

		targetFile, err := os.Create(f.Name)
		if err != nil {
			return err
		}
		defer targetFile.Close()

		data, err := io.ReadAll(rs)
		if err != nil {
			return err
		}

		_, err = targetFile.Write(data)
		if err != nil {
			return err
		}

		rs.Close()
		fmt.Println()
	}

	return nil
}
