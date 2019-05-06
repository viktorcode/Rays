//
//  GameViewController.swift
//  Rays tvOS
//
//  Created by Viktor Chernikov on 16/04/2019.
//  Copyright © 2019 Viktor Chernikov. All rights reserved.
//

import UIKit
import MetalKit

// Our iOS specific view controller
class GameViewController: UIViewController {

    var renderer: Renderer!
    var mtkView: MTKView!

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View of Gameview controller is not an MTKView")
            return
        }

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }

        mtkView.device = defaultDevice
        mtkView.backgroundColor = UIColor.black

		do {
			let newRenderer = try Renderer(withMetalKitView: mtkView)
			renderer = newRenderer
			renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
			mtkView.delegate = renderer
		} catch {
			print("Renderer cannot be initialized : \(error)")
		}
    }
}
