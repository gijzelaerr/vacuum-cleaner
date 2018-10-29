.PHONY: download docker

ALL: download

TARBALL=model.tar.xz                                                                                                    
MODEL=model/$(TARBALL)
URL=http://repo.kernsuite.info/vacuum/$(TARBALL)
DOCKERIMG=gijzelaerr/vacuum-cleaner


model/$(TARBALL):
	cd model && wget $(URL)

model/export.data-00000-of-00001: model/$(TARBALL)
	cd model && tar Jxvfm $(TARBALL)

download: model/export.data-00000-of-00001
	echo "trained model in model/"

docker:
	docker build . -t $(DOCKERIMG)

train: docker
	docker run --runtime=nvidia -it $(DOCKERIMG) vacuum-trainer \
		--input_dir input \
		--output_dir train/docker

clean: docker
	docker run --runtime=nvidia -it $(DOCKERIMG) vacuum-cleaner model/ dirty.fits psf.fits

test: docker
	docker run --runtime=nvidia -it $(DOCKERIMG) vacuum-test \
		--input_dir input \
		--output_dir test/docker \
		--checkpoint train/docker

export_: docker
	docker run --runtime=nvidia -it $(DOCKERIMG) vacuum-export \
		--checkpoint model --output_dir=exported
