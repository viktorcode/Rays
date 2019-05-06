//
//  Renderer.swift
//  Rays iOS
//
//  Created by Viktor Chernikov on 17/04/2019.
//  Copyright © 2019 Viktor Chernikov. All rights reserved.
//

import MetalKit
import MetalPerformanceShaders
import simd
import os

let maxFramesInFlight = 3
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 255) & ~255

let rayStride = 48
let intersectionStride = MemoryLayout<MPSIntersectionDistancePrimitiveIndexCoordinates>.size

enum RendererInitError: Error {
    case noDevice
    case noLibrary
    case noQueue
    case errorCreatingBuffer
}

class Renderer: NSObject, MTKViewDelegate {

    let view: MTKView
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    let accelerationStructure: MPSTriangleAccelerationStructure
    let intersector: MPSRayIntersector

    let vertexPositionBuffer: MTLBuffer
    let vertexNormalBuffer: MTLBuffer
    let vertexColourBuffer: MTLBuffer
    var rayBuffer: MTLBuffer!
    var shadowRayBuffer: MTLBuffer!
    var intersectionBuffer: MTLBuffer!
    let uniformBuffer: MTLBuffer
    let randomBuffer: MTLBuffer
    let triangleMaskBuffer: MTLBuffer

    let rayPipeline: MTLComputePipelineState
    let shadePipeline: MTLComputePipelineState
    let shadowPipeline: MTLComputePipelineState
    let accumulatePipeline: MTLComputePipelineState
    let copyPipeline: MTLRenderPipelineState

    var renderTarget: MTLTexture!
    var accumulationTarget: MTLTexture!

    let semaphore: DispatchSemaphore
    var size: CGSize!
    var randomBufferOffset: Int!
    var uniformBufferOffset: Int!
    var uniformBufferIndex: Int = 0

    var frameIndex: uint = 0

    init(withMetalKitView view: MTKView) throws {
        self.view = view
        guard let device = view.device else { throw RendererInitError.noDevice }
        self.device = device
        os_log("Metal device name is %s", device.name)

        semaphore = DispatchSemaphore(value: maxFramesInFlight)

        // Load Metal
        view.colorPixelFormat = .rgba16Float
        view.sampleCount = 1
        view.drawableSize = view.frame.size
        guard let library = device.makeDefaultLibrary() else { throw RendererInitError.noLibrary }
        self.library = library
        guard let queue = device.makeCommandQueue() else { throw RendererInitError.noQueue }
        self.queue = queue

        // Create pipelines
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        computeDescriptor.computeFunction = library.makeFunction(name: "rayKernel")
        self.rayPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor,
                                                               options: [],
                                                               reflection: nil)
        computeDescriptor.computeFunction = library.makeFunction(name: "shadeKernel")
        self.shadePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor,
                                                                 options: [],
                                                                 reflection: nil)
        computeDescriptor.computeFunction = library.makeFunction(name: "shadowKernel")
        self.shadowPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor,
                                                                  options: [],
                                                                  reflection: nil)
        computeDescriptor.computeFunction = library.makeFunction(name: "accumulateKernel")
        self.accumulatePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor,
                                                                      options: [],
                                                                      reflection: nil)
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.sampleCount = view.sampleCount
        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        self.copyPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)

        // MARK - Create scene
        var vertices = [float3]()
        var normals = [float3]()
        var colours = [float3]()
        var masks = [uint]()

        // Light source
        var transform = Matrix4x4.translation(0, 1, 0) *
            Matrix4x4.scale(0.5, 1.98, 0.5)
        cube(withFaceMask: .positiveY, colour: float3([1, 1, 1]), transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_LIGHT), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // Top, bottom, back
        transform = Matrix4x4.translation(0, 1, 0) * Matrix4x4.scale(2, 2, 2)
        cube(withFaceMask: [.negativeY, .positiveY,.negativeZ], colour: float3([0.725, 0.71, 0.68]), transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // Left wall
        cube(withFaceMask: .negativeX, colour: float3([0.63, 0.065, 0.05]), transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // Right wall
        cube(withFaceMask: .positiveX, colour: float3([0.14, 0.45, 0.091]), transform: transform, inwardNormals: true, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // Short box
        transform = Matrix4x4.translation(0.3275, 0.3, 0.3725) *
            Matrix4x4.rotation(radians: -0.3, axis: float3(0, 1, 0)) *
            Matrix4x4.scale(0.6, 0.6, 0.6)
        cube(withFaceMask: .all, colour: float3([0.725, 0.71, 0.68]), transform: transform, inwardNormals: false, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // Tall box
        transform = Matrix4x4.translation(-0.335, 0.6, -0.29) *
            Matrix4x4.rotation(radians: 0.3, axis: float3(0, 1, 0)) *
            Matrix4x4.scale(0.6, 1.2, 0.6)
        cube(withFaceMask: .all, colour: float3([0.725, 0.71, 0.68]), transform: transform, inwardNormals: false, triangleMask: uint(TRIANGLE_MASK_GEOMETRY), vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)

        // MARK - Create buffers
        // Uniform buffer contains a few small values which change from frame to frame. We will have up to 3
        // frames in flight at once, so allocate a range of the buffer for each frame. The GPU will read from
        // one chunk while the CPU writes to the next chunk. Each chunk must be aligned to 256 bytes on macOS
        // and 16 bytes on iOS.
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight

        // Vertex data should be stored in private or managed buffers on discrete GPU systems (AMD, NVIDIA).
        // Private buffers are stored entirely in GPU memory and cannot be accessed by the CPU. Managed
        // buffers maintain a copy in CPU memory and a copy in GPU memory.
        var options: MTLResourceOptions = []

        #if os(macOS)
        options = .storageModeManaged
        #else // iOS, tvOS
        options = .storageModeShared
        #endif

        // Allocate buffers for vertex positions, colors, and normals. Note that each vertex position is a
        // float3, which is a 16 byte aligned type.
        guard let uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.uniformBuffer = uniformBuffer

        let float2Size = MemoryLayout<float2>.size
        guard let randomBuffer = device.makeBuffer(length: 256 * maxFramesInFlight * float2Size, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.randomBuffer = randomBuffer

        let float3Size = MemoryLayout<float3>.size
        guard let vertexPositionBuffer = device.makeBuffer(bytes: &vertices, length: vertices.count * float3Size, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexPositionBuffer = vertexPositionBuffer

        guard let vertexColourBuffer = device.makeBuffer(bytes: &colours, length: colours.count * float3Size, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexColourBuffer = vertexColourBuffer

        guard let vertexNormalBuffer = device.makeBuffer(bytes: &normals, length: normals.count * float3Size, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.vertexNormalBuffer = vertexNormalBuffer

        let uintSize = MemoryLayout<uint>.size
        guard let triangleMaskBuffer = device.makeBuffer(bytes: &masks, length: masks.count * uintSize, options: options) else {
            throw RendererInitError.errorCreatingBuffer
        }
        self.triangleMaskBuffer = triangleMaskBuffer

        // When using managed buffers, we need to indicate that we modified the buffer so that the GPU
        // copy can be updated
		#if os(macOS)
        if options.contains(.storageModeManaged) {
            vertexPositionBuffer.didModifyRange(0..<vertexPositionBuffer.length)
            vertexColourBuffer.didModifyRange(0..<vertexColourBuffer.length)
            vertexNormalBuffer.didModifyRange(0..<vertexNormalBuffer.length)
            triangleMaskBuffer.didModifyRange(0..<triangleMaskBuffer.length)
        }
		#endif

        // MARK - Create a raytracer for Metal device
        intersector = MPSRayIntersector(device: device)
        intersector.rayDataType = .originMaskDirectionMaxDistance
        intersector.rayStride = rayStride
        intersector.rayMaskOptions = .primitive

        // Create an acceleration structure from our vertex position data
        accelerationStructure = MPSTriangleAccelerationStructure(device: device)
        accelerationStructure.vertexBuffer = vertexPositionBuffer
        accelerationStructure.maskBuffer = triangleMaskBuffer
        accelerationStructure.triangleCount = vertices.count / 3

        accelerationStructure.rebuild()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        self.size = size
        // Handle window size changes by allocating a buffer large enough to contain one standard ray,
        // one shadow ray, and one ray/triangle intersection result per pixel
        let rayCount = Int(size.width * size.height)
        // We use private buffers here because rays and intersection results will be entirely produced
        // and consumed on the GPU
        rayBuffer = device.makeBuffer(length: rayStride * rayCount, options: .storageModePrivate)
        shadowRayBuffer = device.makeBuffer(length: rayStride * rayCount, options: .storageModePrivate)
        intersectionBuffer = device.makeBuffer(length: intersectionStride * rayCount,
                                               options: .storageModePrivate)

        // Create a render target which the shading kernel can write to
        let renderTargetDescriptor = MTLTextureDescriptor()
        renderTargetDescriptor.pixelFormat = .rgba32Float
        renderTargetDescriptor.textureType = .type2D
        renderTargetDescriptor.width = Int(size.width)
        renderTargetDescriptor.height = Int(size.height)
        // Stored in private memory because it will only be read and written from the GPU
        renderTargetDescriptor.storageMode = .private
        // Indicate that we will read and write the texture from the GPU
        renderTargetDescriptor.usage = [.shaderRead, .shaderWrite]

        renderTarget = device.makeTexture(descriptor: renderTargetDescriptor)
        accumulationTarget = device.makeTexture(descriptor: renderTargetDescriptor)
        frameIndex = 0
    }

    func draw(in view: MTKView) {
        // We are using the uniform buffer to stream uniform data to the GPU, so we need to wait until the
        // oldest GPU frame has completed before we can reuse that space in the buffer.
        semaphore.wait()
        // Create a command buffer which will contain our GPU commands
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        // When the frame has finished, signal that we can reuse the uniform buffer space from this frame.
        // Note that the contents of completion handlers should be as fast as possible as the GPU driver may
        // have other work scheduled on the underlying dispatch queue.
        commandBuffer.addCompletedHandler {
            _ in self.semaphore.signal()
        }

        updateUniforms()

        let width = Int(size.width)
        let height = Int(size.height)
        // We will launch a rectangular grid of threads on the GPU to generate the rays. Threads are
        // launched in groups called "threadgroups". We need to align the number of threads to be a multiple
        // of the threadgroup size. We indicated when compiling the pipeline that the threadgroup size would
        // be a multiple of the thread execution width (SIMD group size) which is typically 32 or 64 so 8x8
        // is a safe threadgroup size which should be small to be supported on most devices. A more advanced
        // application would choose the threadgroup size dynamically.
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
        let threadsHeight = threadsPerThreadgroup.height
        let threadsWidth = threadsPerThreadgroup.width
        let threadgroups = MTLSizeMake((width + threadsWidth  - 1) / threadsWidth, (height + threadsHeight - 1) / threadsHeight, 1)
        // First, we will generate rays on the GPU. We create a compute command encoder which will be used
        // to add commands to the command buffer.
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        // Bind buffers needed by the compute pipeline
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
        computeEncoder.setBuffer(rayBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(randomBuffer, offset: randomBufferOffset, index: 2)

        computeEncoder.setTexture(renderTarget, index: 0)
        // Bind the ray generation compute pipeline
        computeEncoder.setComputePipelineState(rayPipeline)
        // Launch threads
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        // End the encoder
        computeEncoder.endEncoding()
        // We will iterate over the next few kernels several times to allow light to bounce around the scene
        for _ in 0..<3 {
            intersector.intersectionDataType = .distancePrimitiveIndexCoordinates
            // We can then pass the rays to the MPSRayIntersector to compute the intersections with our
            // acceleration structure
            intersector.encodeIntersection(commandBuffer: commandBuffer,
                                           intersectionType: .nearest,
                                           rayBuffer: rayBuffer,
                                           rayBufferOffset: 0,
                                           intersectionBuffer: intersectionBuffer,
                                           intersectionBufferOffset: 0,
                                           rayCount: width * height,
                                           accelerationStructure: accelerationStructure)
            // We launch another pipeline to consume the intersection results and shade the scene
            guard let shadeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let buffers = [uniformBuffer, rayBuffer, shadowRayBuffer, intersectionBuffer,
                           vertexColourBuffer, vertexNormalBuffer, randomBuffer, triangleMaskBuffer]
            let offsets: [Int] = [uniformBufferOffset, 0, 0, 0, 0, 0, randomBufferOffset, 0]
            shadeEncoder.setBuffers(buffers, offsets: offsets, range: 0..<8)

            shadeEncoder.setTexture(renderTarget, index: 0)
            shadeEncoder.setComputePipelineState(shadePipeline)
            shadeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            shadeEncoder.endEncoding()

            // We intersect rays with the scene, except this time we are intersecting shadow rays. We only
            // need to know whether the shadows rays hit anything on the way to the light source, not which
            // triangle was intersected. Therefore, we can use the "any" intersection type to end the
            // intersection search as soon as any intersection is found. This is typically much faster than
            // finding the nearest intersection. We can also use MPSIntersectionDataTypeDistance, because we
            // don't need the triangle index and barycentric coordinates.
            intersector.intersectionDataType = .distance
            intersector.encodeIntersection(commandBuffer: commandBuffer,
                                           intersectionType: .any,
                                           rayBuffer: shadowRayBuffer,
                                           rayBufferOffset: 0,
                                           intersectionBuffer: intersectionBuffer,
                                           intersectionBufferOffset: 0,
                                           rayCount: width * height,
                                           accelerationStructure: accelerationStructure)
            // Finally, we launch a kernel which writes the color computed by the shading kernel into the
            // output image, but only if the corresponding shadow ray does not intersect anything on the way
            // to the light. If the shadow ray intersects a triangle before reaching the light source, the
            // original intersection point was in shadow.
            guard let colourEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            colourEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
            colourEncoder.setBuffer(shadowRayBuffer, offset: 0, index: 1)
            colourEncoder.setBuffer(intersectionBuffer, offset: 0, index: 2)

            colourEncoder.setTexture(renderTarget, index: 0)
            colourEncoder.setComputePipelineState(shadowPipeline)
            colourEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            colourEncoder.endEncoding()
        }
        // The final kernel averages the current frame's image with all previous frames to reduce noise due
        // random sampling of the scene.
        guard let denoiseEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        denoiseEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)

        denoiseEncoder.setTexture(renderTarget, index: 0)
        denoiseEncoder.setTexture(accumulationTarget, index: 1)

        denoiseEncoder.setComputePipelineState(accumulatePipeline)
        denoiseEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        denoiseEncoder.endEncoding()

        // Copy the resulting image into our view using the graphics pipeline since we can't write directly
        // to it with a compute kernel. We need to delay getting the current render pass descriptor as long
        // as possible to avoid stalling until the GPU/compositor release a drawable. The render pass
        // descriptor may be nil if the window has moved off screen.
        if let renderPassDescriptor = view.currentRenderPassDescriptor {
            guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
            renderEncoder.setRenderPipelineState(copyPipeline)
            renderEncoder.setFragmentTexture(accumulationTarget, index: 0)
            // Draw a quad which fills the screen
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            renderEncoder.endEncoding()
            // Present the drawable to the screen
            guard let drawable = view.currentDrawable else { return }
            commandBuffer.present(drawable)
        }
        // Finally, commit the command buffer so that the GPU can start executing
        commandBuffer.commit()
    }

    func updateUniforms() {
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        let uniformsPointer = uniformBuffer.contents().advanced(by: uniformBufferOffset)
        let uniforms = uniformsPointer.bindMemory(to: Uniforms.self, capacity: 1)
        uniforms.pointee.camera.position = float3(0, 1, 3.38)

        uniforms.pointee.camera.forward = float3(0, 0, -1)
        uniforms.pointee.camera.right = float3(1, 0, 0)
        uniforms.pointee.camera.up = float3(0, 1, 0)

        uniforms.pointee.light.position = float3(0, 1.98, 0)
        uniforms.pointee.light.forward = float3(0, -1, 0)
        uniforms.pointee.light.right = float3(0.25, 0, 0)
        uniforms.pointee.light.up = float3(0, 0, 0.25)
        uniforms.pointee.light.color = float3(4, 4, 4);

        let fieldOfView = 45.0 * (Float.pi / 180.0)
        let aspectRatio = Float(size.width) / Float(size.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight

        uniforms.pointee.camera.right *= imagePlaneWidth
        uniforms.pointee.camera.up *= imagePlaneHeight

        uniforms.pointee.width = UInt32(size.width)
        uniforms.pointee.height = UInt32(size.height)

        uniforms.pointee.blocksWide = (uniforms.pointee.width + 15) / 16

        frameIndex += 1
        uniforms.pointee.frameIndex = frameIndex

        // For managed storage mode
        #if os(macOS)
        uniformBuffer.didModifyRange(uniformBufferOffset..<uniformBufferOffset + alignedUniformsSize)
        #endif

        randomBufferOffset = 256 * MemoryLayout<float2>.size * uniformBufferIndex
        let float2Pointer = randomBuffer.contents().advanced(by: randomBufferOffset)
        var randoms = float2Pointer.bindMemory(to: float2.self, capacity: 1)
        for _ in 0..<256 {
            randoms.pointee = float2(Float.random(in: 0..<1), Float.random(in: 0..<1))
            randoms = randoms.advanced(by: 1)
        }

        // For managed storage mode
        #if os(macOS)
        randomBuffer.didModifyRange(randomBufferOffset..<randomBufferOffset + 256 * MemoryLayout<float2>.size)
        #endif

        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
}
