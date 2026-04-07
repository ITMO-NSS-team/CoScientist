from ..base import ETLStep
from ..context import ETLContext
from ...domain.entities import ImageInfo
from ...utils.html_processing.images import caption_image


class ImageCaptioningStep(ETLStep):
    
    name = "image_captioning"
    
    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        
        html = ctx.artifact_store.get_html(article_id, "image_filtering")
        manifest_data = ctx.artifact_store.get_metadata(article_id, "image_filtering")
        
        if not manifest_data:
            print(f"[{self.name}] No images to caption for {article_id}")
            ctx.artifact_store.put_html(article_id, self.name, html)
            return
        
        image_infos = [ImageInfo(**data) for data in manifest_data["images"]]
        updated_manifest = []
        images_to_save = {}
        
        for img_info in image_infos:
            if not img_info.is_kept:
                continue
            
            pil_img = ctx.artifact_store.get_image(article_id, "image_filtering", img_info.file_name)
            if pil_img:
                img_info = caption_image(img_info, pil_img, ctx.llm)
                updated_manifest.append(img_info)
                images_to_save[img_info.file_name] = pil_img
        
        if images_to_save:
            ctx.artifact_store.put_images(article_id, self.name, images_to_save)
        
        ctx.artifact_store.put_html(article_id, self.name, html)
        
        manifest_dicts = {"images": [img.model_dump() for img in updated_manifest]}
        ctx.artifact_store.put_metadata(article_id, self.name, manifest_dicts)
