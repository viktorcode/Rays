//
//  Geometry.swift
//  Rays
//
//  Created by Viktor Chernikov on 19/04/2019.
//  Copyright © 2019 Viktor Chernikov. All rights reserved.
//

import simd

struct FaceMask : OptionSet {
    let rawValue: UInt32

    static let negativeX = FaceMask(rawValue: 1 << 0)
    static let positiveX = FaceMask(rawValue: 1 << 1)
    static let negativeY = FaceMask(rawValue: 1 << 2)
    static let positiveY = FaceMask(rawValue: 1 << 3)
    static let negativeZ = FaceMask(rawValue: 1 << 4)
    static let positiveZ = FaceMask(rawValue: 1 << 5)
    static let all: FaceMask = [.negativeX, .negativeY, .negativeZ,
                                .positiveX, .positiveY, .positiveZ]
}

fileprivate func triangleNormal(v0: SIMD3<Float>, v1: SIMD3<Float>, v2: SIMD3<Float>) -> SIMD3<Float> {
    return cross( normalize(v1 - v0), normalize(v2 - v0) )
}

fileprivate func cubeFace(withCubeVertices cubeVertices:[SIMD3<Float>],
                          colour: SIMD3<Float>,
                          index0: Int,
                          index1: Int,
                          index2: Int,
                          index3: Int,
                          inwardNormals: Bool,
                          triangleMask: uint,
                          vertices: inout [SIMD3<Float>],
                          normals: inout [SIMD3<Float>],
                          colours: inout [SIMD3<Float>],
                          masks: inout [uint]) {

    let v0 = cubeVertices[index0]
    let v1 = cubeVertices[index1]
    let v2 = cubeVertices[index2]
    let v3 = cubeVertices[index3]

    var n0 = triangleNormal(v0: v0, v1: v1, v2: v2)
    var n1 = triangleNormal(v0: v0, v1: v2, v2: v3)
    if inwardNormals {
        n0 = -n0
        n1 = -n1
    }

    vertices.append(contentsOf: [v0, v1, v2, v0, v2, v3])
    normals.append(contentsOf: [n0, n0, n0, n1, n1, n1])
    colours.append(contentsOf: [SIMD3<Float>](repeating: colour, count: 6))
    masks.append(contentsOf: [triangleMask, triangleMask])
}

func cube(withFaceMask faceMask: FaceMask,
          colour: SIMD3<Float>,
          transform: matrix_float4x4,
          inwardNormals: Bool,
          triangleMask: uint,
          vertices: inout [SIMD3<Float>],
          normals: inout [SIMD3<Float>],
          colours: inout [SIMD3<Float>],
          masks: inout [uint])
{
    var cubeVertices = [
        SIMD3<Float>(-0.5, -0.5, -0.5),
        SIMD3<Float>( 0.5, -0.5, -0.5),
        SIMD3<Float>(-0.5,  0.5, -0.5),
        SIMD3<Float>( 0.5,  0.5, -0.5),
        SIMD3<Float>(-0.5, -0.5,  0.5),
        SIMD3<Float>( 0.5, -0.5,  0.5),
        SIMD3<Float>(-0.5,  0.5,  0.5),
        SIMD3<Float>( 0.5,  0.5,  0.5),
    ]

    cubeVertices = cubeVertices.map { vertex in
        var transformed = SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
        transformed = transform * transformed
        return SIMD3<Float>(x: transformed.x, y: transformed.y, z: transformed.z)
    }

    if faceMask.contains(.negativeX) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 4, index2: 6, index3: 2,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveX) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 1, index1: 3, index2: 7, index3: 5,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.negativeY) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 1, index2: 5, index3: 4,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveY) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 2, index1: 6, index2: 7, index3: 3,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.negativeZ) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 0, index1: 2, index2: 3, index3: 1,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }

    if faceMask.contains(.positiveZ) {
        cubeFace(withCubeVertices: cubeVertices, colour: colour,
                 index0: 4, index1: 5, index2: 7, index3: 6,
                 inwardNormals: inwardNormals, triangleMask: triangleMask,
                 vertices: &vertices, normals: &normals, colours: &colours, masks: &masks)
    }
}

