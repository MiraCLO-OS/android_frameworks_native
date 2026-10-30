/*
 * Copyright 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS
#include "BlurFilter.h"
#include <algorithm>
#include <SkBlendMode.h>
#include <SkCanvas.h>
#include <SkM44.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/ThreadStateCrashLogger.h>
#include <common/trace.h>
#include <log/log.h>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

const SkString kEffectSource_BlurFilter_MixEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float mixFactor;

    half4 main(float2 xy) {
        return half4(mix(originalInput.eval(xy), blurredInput.eval(xy), mixFactor)).rgb1;
    }
)");

// MiraCLO Liquid Glass. A rounded rectangle (box = centre + half size, radius) of glass: near
// its edge the content behind is pulled inwards along the edge normal following a circular
// bevel, the three colour channels are pulled by slightly different amounts, and a thin rim
// is lit from the top left. Inside the bevel the content is just the blurred input.
const SkString kEffectSource_LiquidGlassEffect(R"(
    uniform shader content;
    uniform float4 box;
    uniform float radius;
    uniform float bevel;
    uniform float strength;
    uniform float dispersion;
    uniform float rimWidth;

    float sdBox(float2 p) {
        float2 q = abs(p - box.xy) - box.zw + radius;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    }

    half4 main(float2 p) {
        float d = sdBox(p);
        float2 n = float2(sdBox(p + float2(1.0, 0.0)) - sdBox(p - float2(1.0, 0.0)),
                          sdBox(p + float2(0.0, 1.0)) - sdBox(p - float2(0.0, 1.0)));
        float nl = length(n);
        n = nl > 0.0001 ? n / nl : float2(0.0);
        float inside = -d;
        float t = clamp(1.0 - inside / bevel, 0.0, 1.0);
        float h = 1.0 - sqrt(max(1.0 - t * t, 0.0));
        float2 off = -n * h * strength * bevel;
        half3 c;
        c.r = content.eval(p + off * (1.0 + dispersion)).r;
        c.g = content.eval(p + off).g;
        c.b = content.eval(p + off * (1.0 - dispersion)).b;
        float lam = dot(n, normalize(float2(-0.55, -0.85)));
        float rim = smoothstep(rimWidth, 0.0, inside);
        float spec = rim * (0.28 + 0.72 * pow(abs(lam), 1.6)) * 0.62;
        float band = t * t * t * 0.10 * max(lam, 0.0);
        c = clamp(c + half3(spec + band), 0.0, 1.0);
        return half4(c, 1.0);
    }
)");

static SkMatrix getShaderTransform(const SkCanvas* canvas, const SkRect& blurRect,
                                   const float scale, const float zoomScale) {
    // 1. Apply the blur shader matrix, which scales up the blurred surface to its real size
    auto matrix = SkMatrix::Scale(scale, scale);
    // 2. Since the blurred surface has the size of the layer, we align it with the
    // top left corner of the layer position.
    matrix.postConcat(SkMatrix::Translate(blurRect.fLeft, blurRect.fTop));
    // 3. Apply the "zoom" effect as an extra scale + translate around the center of the blur.
    if (zoomScale != 1.0f) {
        matrix.postScale(zoomScale, zoomScale);
        matrix.postTranslate(
                blurRect.width() * (1 - zoomScale) / 2.0f,
                blurRect.height() * (1 - zoomScale) / 2.0f);
    }
    // 4. Finally, apply the inverse canvas matrix. The snapshot made in the BlurFilter is in the
    // original surface orientation. The inverse matrix has to be applied to align the blur
    // surface with the current orientation/position of the canvas.
    SkMatrix drawInverse;
    if (canvas != nullptr && canvas->getTotalMatrix().invert(&drawInverse)) {
        matrix.postConcat(drawInverse);
    }
    return matrix;
}

BlurFilter::BlurFilter(RuntimeEffectManager& effectManager, const float maxCrossFadeRadius)
      : mMaxCrossFadeRadius(maxCrossFadeRadius),
        mMixEffect(effectManager.mKnownEffects[kBlurFilter_MixEffect]),
        mLiquidGlassEffect(effectManager.mKnownEffects[kLiquidGlassEffect]) {}

float BlurFilter::getMaxCrossFadeRadius() const {
    return mMaxCrossFadeRadius;
}

sk_sp<SkShader> BlurFilter::makeBlurShader(SkCanvas* canvas, const uint32_t blurRadius,
                                           const float zoomScale, const SkRect& blurRect,
                                           sk_sp<SkImage> blurredImage, sk_sp<SkImage> input) {
    auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, zoomScale);

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    if (blurRadius < mMaxCrossFadeRadius) {
        LOG_THREAD_STATE_AND_CRASH_IF(!input);

        // For sampling Skia's API expects the inverse of what logically seems appropriate. In this
        // case you might expect the matrix to simply be the canvas matrix.
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        if (zoomScale != 1.0f) {
            inputMatrix.preTranslate(
                    blurRect.width() * (1 - zoomScale) / 2.0f,
                    blurRect.height() * (1 - zoomScale) / 2.0f);
            inputMatrix.preScale(zoomScale, zoomScale);
        }

        SkRuntimeShaderBuilder blurBuilder(mMixEffect);
        blurBuilder.child("blurredInput") = blurShader;
        blurBuilder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);

        blurBuilder.uniform("mixFactor") = blurRadius / mMaxCrossFadeRadius;

        return blurBuilder.makeShader();
    }
    return blurShader;
}

void BlurFilter::drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha,
                                const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                sk_sp<SkImage> input) {
    SFTRACE_CALL();

    SkPaint paint;
    paint.setAlphaf(blurAlpha);
    paint.setShader(makeBlurShader(canvas, blurRadius, zoomScale, blurRect, blurredImage, input));

    if (effectRegion.isRect()) {
        if (blurAlpha == 1.0f) {
            paint.setBlendMode(SkBlendMode::kSrc);
        }
        canvas->drawRect(effectRegion.rect(), paint);
    } else {
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
    }
}

void BlurFilter::drawLiquidGlassRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                       const uint32_t blurRadius, const float blurAlpha,
                                       const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                       sk_sp<SkImage> input, const float refraction,
                                       const float bevel, const float dispersion) {
    SFTRACE_CALL();
    const SkRect& r = effectRegion.rect();
    if (!mLiquidGlassEffect || r.isEmpty()) {
        drawBlurRegion(canvas, effectRegion, blurRadius, 1.0f, blurAlpha, blurRect, blurredImage,
                       input);
        return;
    }

    const float halfW = r.width() / 2.0f;
    const float halfH = r.height() / 2.0f;
    const float radius = std::min(effectRegion.radii(SkRRect::kUpperLeft_Corner).x(),
                                  std::min(halfW, halfH));
    const float edge = std::max(1.0f, std::min(bevel, std::min(halfW, halfH)));

    SkRuntimeShaderBuilder glass(mLiquidGlassEffect);
    glass.child("content") =
            makeBlurShader(canvas, blurRadius, 1.0f, blurRect, blurredImage, input);
    glass.uniform("box") = SkV4{r.centerX(), r.centerY(), halfW, halfH};
    glass.uniform("radius") = radius;
    glass.uniform("bevel") = edge;
    glass.uniform("strength") = refraction;
    glass.uniform("dispersion") = dispersion;
    glass.uniform("rimWidth") = std::max(1.5f, edge * 0.09f);

    SkPaint paint;
    paint.setAlphaf(blurAlpha);
    paint.setAntiAlias(true);
    paint.setShader(glass.makeShader());
    canvas->drawRRect(effectRegion, paint);
}

} // namespace skia
} // namespace renderengine
} // namespace android
